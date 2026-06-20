from detector import run_detector

if __name__ == "__main__":
    print("Starting Drowsiness Detection System...")

    try:
        run()
    except Exception as e:
        print("Error:", e)