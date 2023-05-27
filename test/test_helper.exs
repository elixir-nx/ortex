exclude =
  if File.exists?("models/resnet50.onnx") do
    []
  else
    IO.warn(
      """
      skipping resnet50 tests because model is not available.
      Run python/export_resnet.py before for a complete test suite\
      """,
      []
    )

    [:resnet50]
  end

ExUnit.start(exclude: exclude)
