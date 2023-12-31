"""
Data Labeler
"""
import argparse
import os
import re
from pathlib import Path
import logging
import pandas as pd
from metadata_scraper import process_files
from sklearn.preprocessing import LabelEncoder

# =====================================
# Paths
# =====================================
ABSOLUTE_PATH = Path().resolve().parent.parent.parent
RELATIVE_PATH = Path("data/raw/classical_music_files")
RELATIVE_PROCESSED_PATH = Path("data/processed/classical_music_files")
FULL_RAW_PATH = ABSOLUTE_PATH / RELATIVE_PATH
FULL_PROCESSED_PATH = ABSOLUTE_PATH / RELATIVE_PROCESSED_PATH

# =====================================
# Regex patterns
# =====================================
COMPOSER_PATTERN = r"Mozart|Beethoven|Bach"
COMPOSITION_PATTERN = r"Sonata|Concerto|String|Quartet|Quintet|Symphony|Trio|Suite|Prelude|Fugue|\
    Variations|Overture|Rondo|Fantasy|Opera|Divermento|Serenade|Ballet"
COMPOSITION_NUMBER = r"((?:No)(?:.)?\s?\d+)|((?:Nos)(?:.)?\s?\d+\sand\s\d+)|\
    (XC|XL|L?X{0,3})(IX|IV|V?I{0,3})(?=\sin)"
NICKNAME_PATTERN = r"(?<=\").*?(?=\")|(?<=').*?(?=')"
WORKNUMBER_PATTERN = r"((?:K|KV|OP|Opus|BWV)(?:.)?\s?\d+[a-z]?)"
WORKNUMBER_NUMBER_PATTERN = r"((?:No)(?:.)?\s?\d+)"
MOVEMENT_PATTERN = r"\d{1,2}(?:st|nd|rd|th)\sMov(?:ement)?.*|(First|Second|Third|Forth|Fifth|\
    Sixth|Seventh|Eighth|Ninth|Tenth)\sMov(?:ement).*|(IX|IV|V?I{1,3})(\.\s|\s-\s).*|\
        Mov(?:ement)?(?:s)?\s(IX|IV|V?I{1,3}).*|Act\s\d{1,2}(?:.)?.*|((?<=:\s)|(?<=-\s))(Allegro|Andante|Andantino|Adagio|\
        Allegretto|Moderato|Presto|Assai|Menuetto|Rondo|Vivace|Molto|Largo|Larghetto|Romance|Finale|Scherzo|(?:Un\s)?Poco|Grave|Pastorale|Maestoso|\
            Overture|Introduction).*|Variatio\s\d{1,2}.*|\bAria.*\b"
MOVEMENT_NUMBER_PATTERN = r"([0-9][0-9](\.\s|\s-\s).*)"
KEY_PATTERN = (
    r"(?<=in)(?:\s)?\b[A-G]\b(?:-Flat|\sFlat|b|-Sharp|\sSharp|#)?(?:\sMajor|\sMinor)?"
)
INSTRUMENT_PATTERN = r"\b(Piano|Keyboard|Organ|Guitar|Violin|Viola|Cello|\
    Double Bass|Piccolo|(?<!Magic\s)Flute|Oboe|Clarinet|Bassoon|Trumpet|\
    Horn|Trombone|Tuba|Saxophone|Timpani|Harp|Recorder|Bagpipes|Ukulele)(?:s)?\b"


def extract_composer(file, file_path):
    """
    Extracts composer information from the file metadata.

    """
    # Get all possible entires from the file to search for the composer.
    entires = [file.artist, file.title, file.album]
    for entry in entires:
        composer_result = re.search(COMPOSER_PATTERN, entry, re.IGNORECASE)
        if composer_result is not None:
            return composer_result.group(0).lower().capitalize()
    logging.debug(
        "No composer found for %s, falling back to searching in file path", file.title
    )
    composer_result = re.search(COMPOSER_PATTERN, file_path, re.IGNORECASE)
    return composer_result.group(0).lower().capitalize()


def extract_composition(title):
    """
    Extracts the composition from the title entry.

    """
    composition_result = re.findall(COMPOSITION_PATTERN, title, re.IGNORECASE)
    if composition_result is not None:
        matches = list(map(lambda s: s.capitalize(), composition_result))
        matches = list(dict.fromkeys(matches))
        return ", ".join(matches)
    logging.debug("No composition found for %s", title)
    return None


def extract_composition_no(title):
    """
    Extracts the composition number from the title entry.

    """
    composition_result = re.search(COMPOSITION_PATTERN, title, re.IGNORECASE)
    start_position = 0
    if composition_result:
        start_position = composition_result.end()
    else:
        instrument_result = re.search(INSTRUMENT_PATTERN, title, re.IGNORECASE)
        if instrument_result:
            start_position = instrument_result.end()
    composition_no_result = re.search(
        COMPOSITION_NUMBER, title[start_position:], re.IGNORECASE
    )
    if composition_no_result is not None:
        return composition_no_result.group(0)
    logging.debug("No composition number found for %s", title)
    return None


def extract_nickname(title):
    """
    Extracts the nickname of the work from the title entry.

    """
    nickname_result = re.search(NICKNAME_PATTERN, title, re.IGNORECASE)
    if nickname_result is not None:
        return nickname_result.group(0)
    logging.debug("No nickname found for %s", title)
    return None


def extract_workno(title):
    """
    Extract work number (opus) information from the file metadata.

    """
    # Get all possible entires from the file to search for the work number.
    result = re.search(WORKNUMBER_PATTERN, title, re.IGNORECASE)
    if result is not None:
        number_result = re.search(
            WORKNUMBER_NUMBER_PATTERN, title[result.end() :], re.IGNORECASE
        )
        if number_result:
            return f"{result.group(0)}, {number_result.group(0)}"
        return result.group(0)
    logging.debug("No work number found for %s", title)
    return None


def extract_movement(title):
    """
    Find all movements that begins with a number from the title entry.

    First filter all results that contain the work number then find the movement.
    """
    title_search = title
    workno_last_match = list(re.finditer(WORKNUMBER_PATTERN, title, re.IGNORECASE))
    if workno_last_match:
        start_position = workno_last_match[-1].end()
        title_search = title[start_position:]
    movement_result = re.search(MOVEMENT_NUMBER_PATTERN, title_search, re.IGNORECASE)
    if movement_result is not None:
        return movement_result.group(0)
    movement_result = re.search(MOVEMENT_PATTERN, title, re.IGNORECASE)
    if movement_result is not None:
        return movement_result.group(0)
    logging.debug("No movement found for %s", title)
    return None


def extract_key(title):
    """
    Extracts key information from the title.

    """
    key_result = re.search(KEY_PATTERN, title, re.IGNORECASE)
    if key_result is not None:
        return key_result.group(0)
    logging.debug("No key found for %s", title)
    return None


def extract_instruments(title):
    """
    Extracts instrument information (if applicable) from the title

    """
    result = re.findall(INSTRUMENT_PATTERN, title, re.IGNORECASE)
    if result is not None:
        matches = list(map(lambda s: s.capitalize(), result))
        matches = list(dict.fromkeys(matches))
        return ", ".join(matches)
    logging.debug("No instrument(s) found for %s", title)
    return None


def label_data(args):
    """
    Label the audio files
    """
    columns = [
        "path",
        "filename",
        "title",
        "composer",
    ]
    files_df = pd.DataFrame(columns=columns)
    for _, (path, file) in enumerate(process_files()):
        # If the metadata of an audio file is completely empty, we will skip it
        if all(v is None for v in [file.artist, file.title, file.album]):
            logging.debug("Skipping %s because its metadata is empty", str(path.name))
            continue
        composer = extract_composer(file, str(path.parent))
        new_row = {
            "path": str(path.parent),
            "filename": str(path.name),
            "title": file.title,
            "composer": composer,
        }
        if args.composition:
            composition = extract_composition(file.title)
            composition_number = extract_composition_no(file.title)
            columns.extend(["composition", "composition_number"])
            new_row["composition"] = composition
            new_row["composition_number"] = composition_number
        if args.nickname:
            nickname = extract_nickname(file.title)
            columns.append("nickname")
            new_row["nickname"] = nickname
        if args.worknumber:
            worknumber = extract_workno(file.title)
            columns.append("workno")
            new_row["workno"] = worknumber
        if args.key:
            key = extract_key(file.title)
            columns.append("key")
            new_row["key"] = key
        if args.movement:
            movement = extract_movement(file.title)
            columns.append("movement")
            new_row["movement"] = movement
        if args.instruments:
            instruments = extract_instruments(file.title)
            columns.append("instruments")
            new_row["instruments"] = instruments
        files_df = pd.concat([files_df, pd.DataFrame([new_row])], ignore_index=True)
    files_df = files_df.fillna("")
    labeler = LabelEncoder()
    composer_enc = labeler.fit_transform(files_df.loc[:, "composer"])
    files_df["composer_enc"] = composer_enc
    logging.info("Processed %d music files", len(files_df))
    return files_df


def save_csv(dataframe):
    """
    Saves the data as a CSV
    """
    try:
        dataframe.to_csv(os.path.join(FULL_PROCESSED_PATH, "metadata.csv"), index=False)
        logging.info(
            "Metadata saved successfully to %s/metadata.csv", FULL_PROCESSED_PATH
        )
    except OSError as error:
        logging.error("Saving metadata failed! %s", str(error))


def to_bool(flag):
    """
    Converts the T/F, 1/0, or Y/N flag into a boolean
    """
    if flag.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if flag.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def process_cmdline_options():
    """
    Parses all the command line options
    """
    parser = argparse.ArgumentParser(
        description="Automatically label all music files and store the results as a CSV file."
    )
    parser.add_argument(
        "--composition",
        default=False,
        action="store_true",
        help="Adds the composition title and number to the output.",
    )
    parser.add_argument(
        "--nickname",
        default=False,
        action="store_true",
        help="Adds the nickname of the work to the output (if applicable).",
    )
    parser.add_argument(
        "--worknumber",
        default=False,
        action="store_true",
        help="Adds the number of the work to the output (if applicable).",
    )
    parser.add_argument(
        "--key",
        default=False,
        action="store_true",
        help="Adds the key of the work to the output (if applicable)",
    )
    parser.add_argument(
        "--movement",
        default=False,
        action="store_true",
        help="Adds the movement of the work to the output (if applicable)",
    )
    parser.add_argument(
        "--instruments",
        default=False,
        action="store_true",
        help="Adds the instrument(s) of the work to the output (if applicable)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO
    )
    logging.info("Loading music data labeler")
    arguments = process_cmdline_options()
    df = label_data(arguments)
    save_csv(df)
