import time
import subprocess

RUN_INTERVAL = 60  # seconds


def run_pipeline():

    print("\nRUNNING MASTER PIPELINE\n")

    subprocess.run(
        ["python", "master_pipeline.py"]
    )


def run_auto():

    print("\nRUNNING AUTO ENGINE\n")

    subprocess.run(
        ["python", "automation_engine/auto_engine.py"]
    )


def start_loop():

    print("\nAUTO LOOP STARTED\n")

    while True:

        run_pipeline()

        run_auto()

        print(f"\nWaiting {RUN_INTERVAL} seconds...\n")

        time.sleep(RUN_INTERVAL)


if __name__ == "__main__":

    start_loop()