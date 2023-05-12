model = Ortex.load("./models/stability-lm-3b/stability-lm-tuned-3b.onnx")

prompt = "<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
<|USER|>How are you feeling? <|ASSISTANT|>
"

{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-3b")
{:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, prompt)

input = Nx.tensor([Tokenizers.Encoding.get_ids(encoding)])
mask = Nx.tensor([Tokenizers.Encoding.get_attention_mask(encoding)])

defmodule M do
  def generate(_model, input, _mask, 500) do
    input
  end

  def generate(model, input, mask, iter) do
    [output | _] =
      Ortex.run(model, {
        input,
        mask
      })
      |> Tuple.to_list()

    x = output |> Nx.backend_transfer() |> Nx.argmax(axis: 2)
    last = x[[.., -1]] |> Nx.new_axis(0)
    IO.inspect(last[0][0] |> Nx.to_number)

    case Enum.member?([50278, 50279, 50277, 1, 0], last[0][0] |> Nx.to_number) do
      true ->
        input

      false ->
        generate(
          model,
          Nx.concatenate([input, last], axis: 1),
          Nx.concatenate([mask, Nx.tensor([[1]])], axis: 1),
          iter + 1
        )
    end
  end
end

result = M.generate(model, input, mask, 0)
IO.inspect(result)

IO.inspect(
  Tokenizers.Tokenizer.decode(
    tokenizer,
    result
    |> Nx.backend_transfer()
    |> Nx.to_batched(1)
    |> Enum.map(&Nx.to_flat_list/1)
  )
)
