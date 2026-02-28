import argparse
import logging
from app.scripts.RH_Dataset import import_hr_data
from app.scripts.seed_demo_data import seed_demo_data

def main():
    parser = argparse.ArgumentParser(description="run dataset import and demo seed")
    parser.add_argument("--skip-import", action="store_true", help="skip HR CSV import")
    parser.add_argument("--skip-import", action="store_true", help="skip demo seed data")
    args = parser.parse_args()

    if not args.skip_import:
        import_hr_data()

    if not args.skip_demo:
        seed_demo_data()

if __name__ == "main":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main()
