import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python update_metadata_version.py <version>")
    sys.exit(1)

version = sys.argv[1].lstrip("v")  # remove leading 'v' if the tag has it

metadata_file = Path("metadata.txt")

content = metadata_file.read_text()

# Replace version in metadata.txt
new_content = re.sub(r"^version=.*", f"version={version}",
                     content, flags=re.MULTILINE)
metadata_file.write_text(new_content)

print(f"Updated metadata.txt version to {version}")
