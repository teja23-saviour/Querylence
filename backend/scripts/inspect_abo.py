
import gzip
import tarfile
from pathlib import Path

ARCHIVE_PATH = Path("data/raw/abo-listings.tar")


def main():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Archive not found: {ARCHIVE_PATH.resolve()}"
        )

    with tarfile.open(ARCHIVE_PATH, mode="r") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.startswith("listings/metadata/")
            and member.name.endswith(".json.gz")
        ]

        if not members:
            raise RuntimeError("No compressed listing files found.")

        print(f"Metadata files found: {len(members)}")
        print(f"Inspecting: {members[0].name}\n")

        raw_file = archive.extractfile(members[0])
        if raw_file is None:
            raise RuntimeError("Could not read the metadata file.")

        with raw_file:
            with gzip.GzipFile(fileobj=raw_file) as compressed:
                sample = compressed.read(5000)

        print("----- DECOMPRESSED JSON SAMPLE -----")
        print(sample.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
