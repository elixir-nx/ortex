defmodule Ortex.MixProject do
  use Mix.Project

  @version "0.1.11"
  @source_url "https://github.com/elixir-nx/ortex"

  def project do
    [
      app: :ortex,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      # Docs
      name: "Ortex",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: [
        main: "readme",
        extras: ["README.md", "CHANGELOG.md"]
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
      # Rustler is required only for source-compiled builds (when
      # `ORTEX_BUILD=true` is set or no precompiled artifact exists for
      # the target). Made optional so consumers using precompiled NIFs
      # don't need a Rust toolchain on PATH. Kept at the original
      # `~> 0.27` constraint that the source-build path was tested
      # against — bumping the floor showed no benefit and risked
      # changing build-output paths in ways `Ortex.Util.copy_ort_libs/0`
      # couldn't follow.
      {:rustler, "~> 0.27", optional: true},
      {:rustler_precompiled, "~> 0.8"},
      {:nx, "~> 0.6"},
      {:tokenizers, "~> 0.4", only: :dev},
      {:ex_doc, "0.29.4", only: :dev, runtime: false},
      {:exla, "~> 0.6", only: :dev},
      {:torchx, "~> 0.6", only: :dev}
    ]
  end

  defp package do
    [
      files: ~w(
        lib
        .formatter.exs
        mix.exs
        README*
        CHANGELOG*
        LICENSE*
        checksum-*.exs
        native/ortex/src/
        config/config.exs
        native/ortex/Cargo.lock
        native/ortex/Cargo.toml
        native/ortex/.cargo/config.toml
      ),
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      description: "ONNX Runtime bindings for Elixir"
    ]
  end
end
