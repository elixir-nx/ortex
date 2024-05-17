defmodule Inference do

  def id_to_label(id) do
    {:ok, config_json} = File.read("./models/distilbert-onnx/config.json")
    {:ok, %{"id2label" => id2label}} = Jason.decode(config_json)
    Map.get(id2label, to_string(id))
  end
  

  def run() do
    model = Ortex.load("./models/distilbert-onnx/model.onnx")
    text = "the movie had a lot of nuance and interesting artistic choices, would like to see more support in the industry for these types of productions"

    {:ok, tokenizer} = Tokenizers.Tokenizer.from_file("./models/distilbert-onnx/tokenizer.json")
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)

    input = Nx.tensor([Tokenizers.Encoding.get_ids(encoding)])
    mask = Nx.tensor([Tokenizers.Encoding.get_attention_mask(encoding)])

    {output} = Ortex.run(model, {input, mask})

    IO.inspect(output)

    IO.inspect(
      output
      |> Nx.backend_transfer()
      |> Nx.argmax()
      |> Nx.to_number()
      |> id_to_label()
    )
  end
end

Inference.run()
