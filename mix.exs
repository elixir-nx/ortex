defmodule Ortex.MixProject do
  use Mix.Project

  def project do
    [
      app: :ortex,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Docs
      name: "Ortex",
      source_url: "https://github.com/relaypro-open/ortex",
      homepage_url: "http://github.com/relaypro-open/ortex",
      docs: [
        main: "readme",
        extras: ["README.md"]
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.26.0"},
      {:useful, "~>1.11.0"},
      {:nx, "~>0.5.3"},
      # {:dialyxir, "~>1.3.0", only: [:dev], runtime: false},
      {:tokenizers, "~> 0.3.0"},
      {:ex_doc, "0.29.4", only: :dev, runtime: false},
      {:axon_onnx, "~>0.4.0"},
      {:exla, "~> 0.5"},
      {:torchx, "~> 0.5"}
    ]
  end
end
