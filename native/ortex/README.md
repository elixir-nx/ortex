# NIF for Elixir.Ortex.Native

## To build the NIF module:

- Your NIF will now build along with your project.

## To load the NIF:

```elixir
defmodule Ortex do
  use Rustler, otp_app: :ortex, crate: "ortex"

  # When your NIF is loaded, it will override this function.
  def add(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
end
```

## Examples

[This](https://github.com/rusterlium/NifIo) is a complete example of a NIF written in Rust.

## Docs
To build the documentation for this locally, run

```shell
cargo doc --open --no-deps --document-private-items
```
