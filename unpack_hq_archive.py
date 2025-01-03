import sys
import tarfile
from argparse import ArgumentParser
from pathlib import Path
from tarfile import TarInfo

import pandas as pd


def is_stdout_file(fileinfo: TarInfo):
    return fileinfo.name.endswith(".stdout")


def eprint(*values):
    print(*values, file=sys.stderr, flush=True)


def abort(message):
    eprint(str(message))
    sys.exit(1)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("filename", type=Path)
    parser.add_argument("--input-format", choices=['csv', 'json'], default='json')
    return parser.parse_args()


def main():
    args = parse_args()
    filename: Path = args.filename
    input_format = args.input_format

    try:
        rows = []
        with tarfile.open(filename) as archive:
            for member in filter(is_stdout_file, archive):
                file = archive.extractfile(member)
                try:
                    if input_format == 'csv':
                        rows.append(pd.read_csv(file))
                    else:
                        rows.append(pd.read_json(file, typ='series').to_frame().transpose())
                except ValueError as err:
                    eprint(f"warning: {file.name}: {str(err)}")
                except pd.errors.EmptyDataError as err:
                    eprint(f"warning: {file.name}: {str(err)}")
    except FileNotFoundError as err:
        abort(f"error: {err.strerror}: {err.filename}")
    except tarfile.ReadError:
        abort(f"error: Unable to open tar archive")
    else:
        df = pd.concat(rows).convert_dtypes()
        df.to_csv(filename.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
