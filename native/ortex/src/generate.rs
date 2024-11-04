use anyhow::{anyhow, Result};

use std::collections::hash_map::HashMap;

#[derive(Debug)]
struct BeamCandidate {
    tokens: Vec<u32>,
    score: f32,
}

impl BeamCandidate {
    fn new(tokens: Vec<u32>) -> BeamCandidate {
        Self { tokens, score: 0.0 }
    }
    fn len(&self) -> usize {
        return self.tokens.len();
    }
    fn push(&mut self, token_and_score: (u32, f32)) {
        let (token, score) = token_and_score;
        self.tokens.push(token);
        self.score = self.score + score;
    }
}

impl Clone for BeamCandidate {
    fn clone(&self) -> BeamCandidate {
        BeamCandidate {
            tokens: self.tokens.clone(),
            score: self.score.clone(),
        }
    }
}
impl Ord for BeamCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

impl Eq for BeamCandidate {}

impl PartialOrd for BeamCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl PartialEq for BeamCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.tokens.eq(&other.tokens)
    }
}

fn _topk<K>(x: Vec<K>, k: usize) -> Result<Vec<(usize, K)>>
where
    K: Clone + PartialOrd,
{
    let l = x.len() as usize;

    if k > l {
        // TODO: make this an error for a result
        return Err(anyhow!("This is an error"));
    }

    let mut a: Vec<_> = x.into_iter().enumerate().collect();
    a.sort_by(|(_i1, s1), (_i2, s2)| s2.partial_cmp(s1).unwrap());
    Ok(a.into_iter().take(k).collect())
}

// TODO: Needs some work to speed up. Explore Candle-Custom Opts and Candle-Ext
fn topk_2d<K>(batched_log_probs: Vec<Vec<K>>, k: usize) -> Vec<Vec<(usize, K)>>
where
    K: PartialOrd + Clone,
{
    let mut output = Vec::new();
    for batch in batched_log_probs.iter() {
        let mut enumerated_tokens: Vec<(usize, &K)> = batch.iter().enumerate().collect();
        enumerated_tokens.sort_by(|(_i1, s1), (_i2, s2)| s2.partial_cmp(s1).unwrap());
        let batch_topk: Vec<&(usize, &K)> = enumerated_tokens.iter().take(k).collect();
        // copy the k references
        let copied_batch_topk: Vec<(usize, K)> = batch_topk
            .iter()
            .cloned()
            .map(|(s, k)| (*s, (**k).clone()))
            .collect();
        output.push(copied_batch_topk);
    }
    output
}

pub fn beam_search(
    encoder: ResourceArc<OrtexModel>,
    decoder: ResourceArc<OrtexModel>,
    audio: ResourceArc<OrtexTensor>,
    start_token: u32,
    lang_token: u32,
    transcribe_token: u32,
    no_timestamps_token: u32,
    eot_token: u32,
    num_beams: usize,
    constraints: Vec<String>,
) -> Result<Vec<u32>> {
    println!("Beam search beginning");
    let overall = std::time::SystemTime::now();
    // build beam search stuff
    let mut completed_sequences: Vec<BeamCandidate> = Vec::new();

    // TODO: these are fixed - can we make them an array?
    let initial_tokens = utils::init_tokens(&tokenizer)?;
    // TODO: num-beams is fixed and we can set a max length. Also an array?
    let mut beams: Vec<BeamCandidate> = (0..num_beams)
        .map(|_| BeamCandidate::new(initial_tokens.clone()))
        .collect();
    let eot_id = utils::token_id(tokenizer, m::EOT_TOKEN)?;

    let mut expanded_shape = encoder_output.shape().clone().into_dims();
    expanded_shape[0] = num_beams;
    println!("{:?}", expanded_shape);
    // to check; does expand copy or just change shape?
    let batched_encoder_output =
        encoder_output.expand(Shape::from_dims(expanded_shape.as_slice()))?;

    // begin beamin
    for i in 1..1000 {
        let start = std::time::SystemTime::now();
        // input-prep
        let seq_len = beams[0].len();
        let beam_shape = Shape::from_dims(&[num_beams, seq_len]);
        let beam_tokens: Vec<Vec<u32>> = beams
            .iter()
            .map(|candidate| candidate.tokens.clone())
            .collect();
        let flattened_beams: Vec<u32> = beam_tokens.into_iter().flatten().collect();
        let beam_t = Tensor::from_slice(&flattened_beams[..], beam_shape, device)?;

        let preprocess_time = start.elapsed().unwrap();

        // forward pass + processing
        // [bsz; seq_len; vocab_size]
        let ys = model
            .decoder
            .forward(&beam_t, &batched_encoder_output, false)?;
        // [bsz; 1; vocab_size]
        let logits = model.decoder.final_linear(&ys.i((.., seq_len - 1..))?)?;
        // [bsz; vocab_size]
        let logits = logits.i((.., 0, ..))?;
        let logprobs = log_softmax(&logits, candle_core::D::Minus1)?;

        let forward_time = start.elapsed().unwrap();

        // for beam we started with, and per candidate, create each possible beam
        let beam_logprobs_v: Vec<Vec<f32>> = logprobs.to_vec2()?;
        let convert_to_2d_vec_time = start.elapsed().unwrap();

        // masking! first, build a masked version of the outputs, and generate valid continuations
        // using the beams we've currently built
        let mut masked_beam_logprobs =
            vec![vec![f32::NEG_INFINITY; beam_logprobs_v[0].len()]; beam_logprobs_v.len()];
        println!("Getting continuations from beams: {:?}", &beams);
        let valid_continuations = get_valid_continuations(&beams, &continuations);
        println!("Continuations: {:?}", &valid_continuations);

        // NOTE: we assume that there are far fewer valid continuations than there outputs;
        // therefore, we iterate through the continutations and unmask rather than checking
        // each in the unmasked to see if it should be masked
        for (beam_idx, continuation) in valid_continuations.iter().enumerate() {
            for continuation_idx in continuation.into_iter() {
                println!("unmasking {}", continuation_idx);
                masked_beam_logprobs[beam_idx][*continuation_idx as usize] =
                    beam_logprobs_v[beam_idx][*continuation_idx as usize]
            }
        }

        // Grab topk from masks
        let topk_beam_candidates: Vec<Vec<(usize, f32)>> =
            topk_2d(masked_beam_logprobs, num_beams * 2);

        let topk_time = start.elapsed().unwrap();
        // beam processing
        let mut candidate_beams = Vec::new();

        // iterate through each top beam and create a new candidate beam.
        // Then check to see if it's an EOT, and if so add it to the finished
        // Otherwise, add it to a candidate list
        for (beam_idx, beam_candidates) in topk_beam_candidates.iter().enumerate() {
            let curr_beam = unsafe { beams.get_unchecked(beam_idx) };
            for (candidate_token, candidate_score) in beam_candidates.iter() {
                let candidate_token: u32 = candidate_token.clone().try_into()?;
                let mut curr_beam = curr_beam.clone();
                curr_beam.push((candidate_token, candidate_score.clone()));
                if candidate_token == eot_id {
                    // println!("COMPLETED candidate beam: {:?}", next_candidate);
                    completed_sequences.push(curr_beam);
                    // make sure it doesn't already exist in there.
                } else {
                    // println!("adding candidate beam: {:?}", next_candidate);
                    candidate_beams.push(curr_beam);
                }
            }
        }

        let beam_processing_time = start.elapsed().unwrap();

        // NOTE: Not super concerned with performance here; this is beam_size^2 * seq-len,
        // where beam-size and seq len are typically < 100;

        // sort candidates, remove any duplicates
        candidate_beams.sort();
        candidate_beams.dedup_by_key(|c| c.tokens.clone());
        candidate_beams.reverse();

        // truncate to only the top n_beam candidates, or if not enough candidates,
        // expand the best candidate however many times to get num_beams beams
        candidate_beams.resize(num_beams, candidate_beams[0].clone());

        // sort/remove any duplicates
        completed_sequences.sort();
        completed_sequences.dedup_by_key(|c| c.tokens.clone());
        completed_sequences.reverse();

        let sorting_time = start.elapsed().unwrap();

        // now, we'll check to see if we've found num-beams. if we have,
        // also check to see if there are any promising candidates yet to
        // be completed.
        let default = BeamCandidate {
            tokens: Vec::new(),
            score: f32::NEG_INFINITY,
        };
        let best_completed = completed_sequences.get(0).unwrap_or(&default);
        let best_candidate = candidate_beams.get(0).unwrap_or(&default);
        // TODO: Check if all remaining beams are neg-inf (e.g., masked out)
        if completed_sequences.len() >= num_beams && best_completed.score > best_candidate.score {
            break;
        }

        let completed_time = start.elapsed().unwrap();
        println!(
            "{}, {:?}, {:?}, {:?} {:?}, {:?}, {:?}, {:?} {:?}",
            i,
            start,
            preprocess_time,
            forward_time,
            convert_to_2d_vec_time,
            topk_time,
            beam_processing_time,
            sorting_time,
            completed_time
        );
        beams = candidate_beams;
    }

    let selected_len = &completed_sequences[0].len();
    let end_time = overall.elapsed().unwrap();

    println!(
        "Generated {} tokens in {:?}, ({} tokens/sec)",
        selected_len,
        end_time,
        (*selected_len as f32) / (end_time.as_secs_f32())
    );
    println!(
        "{}",
        tokenizer
            .decode(&completed_sequences[0].tokens, false)
            .unwrap()
    );

    Ok(Vec::new())
}

fn get_valid_continuations(
    beams: &Vec<BeamCandidate>,
    continuations: &HashMap<Vec<u32>, Vec<u32>>,
) -> Vec<Vec<u32>> {
    beams
        .iter()
        .map(|beam| {
            continuations
                .get(&beam.tokens)
                .unwrap_or(&vec![50257])
                .clone()
        })
        .collect()
}

fn generate_continations(
    commands: &Vec<String>,
    tokenizer: &Tokenizer,
) -> HashMap<Vec<u32>, Vec<u32>> {
    let mut continuations: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
    for command in commands.iter() {
        // TODO: fix this to reference m::START_OF_TRANSCRIPT..., etc.
        let command =
            "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>".to_string() + command;
        let command = command + "<|endoftext|>";

        let tokens = tokenizer.encode(command.clone(), false).unwrap();
        let tokens = tokens.get_ids().to_vec();
        let mut prefix = Vec::new();
        prefix.push(tokens.first().unwrap().clone());
        let mut tokens_iter = tokens.iter();
        tokens_iter.next();

        for token in tokens_iter {
            if let Some(next) = continuations.get_mut(&prefix) {
                if !next.contains(token) {
                    next.push(token.clone());
                }
            } else {
                continuations.insert(prefix.clone(), vec![token.clone()]);
            }
            prefix.push(token.clone());
        }
        println!("{:?}", tokenizer.decode(prefix.as_slice(), false));
    }
    continuations
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (mut model, tokenizer, config) = utils::init_model("openai/whisper-medium", &device)?;
    let commands = vec!["Call Sally", "Call Mindy", "Switch to Slot ops"];
    let commands = commands.iter().map(|i| i.to_string()).collect();
    let continuations = generate_continations(&commands, &tokenizer);
    for continuation in continuations.iter() {
        println!("{:?}", continuation);
    }

    beam_search(&mut model, &config, &device, &tokenizer, 3, continuations)?;
    greedy_search(&mut model, &config, &device, &tokenizer)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils;
    use crate::{generate_continations, get_valid_continuations, BeamCandidate};
    use anyhow::Result;
    use candle_core::Device;

    fn get_dummy_commands() -> Vec<String> {
        let v = vec!["Call Sally", "Call Mindy"];
        v.iter().map(|i| i.to_string()).collect()
    }

    #[test]
    fn test_continuation_map() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (_model, tokenizer, _config) = utils::init_model("openai/whisper-medium", &device)?;
        let commands = get_dummy_commands();
        let continuations = generate_continations(&commands, &tokenizer);
        let tokenized_commands: Vec<Vec<u32>> = commands
            .iter()
            .map(|command| {
                let command = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
                    .to_string()
                    + command;
                let command = command + "<|endoftext|>";
                let tokens = tokenizer.encode(command[..].to_string(), false).unwrap();
                let tokens = tokens.get_ids().to_vec();
                println!("{:?} -> {:?}", command, tokens);
                tokens
            })
            .collect();

        let truncated_candidates: Vec<BeamCandidate> = tokenized_commands
            .iter()
            .map(|tokens| {
                let tokens = tokens[0..5].to_vec();
                let score = 0.0f32;
                BeamCandidate { tokens, score }
            })
            .collect();

        let beamed_next_possible_tokens =
            get_valid_continuations(&truncated_candidates, &continuations);

        for (i, next_possible_tokens) in beamed_next_possible_tokens.iter().enumerate() {
            let target_token = tokenized_commands[i][5];
            assert!(next_possible_tokens.contains(&target_token));
        }
        Ok(())
    }
}
