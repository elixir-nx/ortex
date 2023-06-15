defmodule Ortex.MixProject do
  use Mix.Project

  def project do
    [
      app: :ortex,
      version: "0.1.5",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Docs
      name: "Ortex",
      source_url: "https://github.com/elixir-nx/ortex",
      homepage_url: "http://github.com/elixir-nx/ortex",
      docs: [
        main: "readme",
        extras: ["README.md"]
      ],
      package: package()
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
      {:rustler, "~> 0.28.0"},
      {:nx, "~>0.5.3"},
      {:tokenizers, "~> 0.3.0", only: :dev},
      {:ex_doc, "0.29.4", only: :dev, runtime: false},
      {:axon_onnx, "~>0.4.0", only: :dev},
      {:exla, "~> 0.5", only: :dev},
      {:torchx, "~> 0.5", only: :dev}
    ]
  end

  defp package do
    [
      files: ~w(lib .formatter.exs mix.exs README* LICENSE* native/ortex/src/
        native/ortex/Cargo.lock native/ortex/Cargo.toml native/ortex/.cargo/config.toml),
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/elixir-nx/ortex"},
      description: "ONNX Runtime bindings for Elixir"
    ]
  end
end
