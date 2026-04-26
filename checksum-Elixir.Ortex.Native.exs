# SHA-256 checksums for the precompiled NIF artifacts published to
# the project's GitHub releases. RustlerPrecompiled refuses to
# download any artifact whose checksum is not pinned here — this
# file is the integrity boundary between consumers and the release
# tarballs.
#
# This file is auto-populated as part of the release process; do not
# hand-edit. To regenerate after a tag's CI run finishes and the
# draft release has been promoted:
#
#     ORTEX_BUILD=true mix rustler_precompiled.download \
#         Ortex.Native --all
#
# (The ORTEX_BUILD=true is needed on the first run when the file is
# empty — without it, mix can't compile lib/ortex/native.ex because
# the precompiled-NIF download path tries to verify against an empty
# checksum map. Setting it forces source build during the bootstrap
# compile, after which the download task runs and writes the file.)
#
# Commit the result before running mix hex.publish.
%{}
