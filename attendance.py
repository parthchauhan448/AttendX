import argparse
import json
import shutil
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "attendance_data"
FACES_DIR = DATA_DIR / "faces"
PEOPLE_FILE = DATA_DIR / "people.json"
MODEL_FILE = DATA_DIR / "recognizer.yml"
ATTENDANCE_FILE = DATA_DIR / "attendance.xlsx"
FACE_SIZE = (200, 200)


def load_people():
    if not PEOPLE_FILE.exists():
        return {}
    return json.loads(PEOPLE_FILE.read_text(encoding="utf-8"))


def get_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV could not load its face detection model.")
    return detector


def find_faces(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    return gray, faces


def register_person(person_id, name, sample_count, replace=False):
    if not person_id.isdigit() or int(person_id) < 1:
        raise ValueError("ID must be a positive whole number.")
    person_id = str(int(person_id))
    if not name.strip():
        raise ValueError("Name cannot be empty.")

    people = load_people()
    if person_id in people and not replace:
        raise ValueError(f"ID {person_id} is already registered. Use --replace to recapture it.")

    person_dir = FACES_DIR / person_id
    capture_dir = FACES_DIR / f".{person_id}_capture"
    detector = get_detector()
    if capture_dir.exists():
        shutil.rmtree(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        shutil.rmtree(capture_dir)
        raise RuntimeError("Could not open the webcam. Check that it is connected and available.")

    saved = 0
    try:
        while saved < sample_count:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            gray, faces = find_faces(frame, detector)
            display = frame.copy()
            for x, y, width, height in faces:
                cv2.rectangle(display, (x, y), (x + width, y + height), (50, 210, 80), 2)

            if len(faces) == 1:
                x, y, width, height = faces[0]
                face = cv2.equalizeHist(gray[y : y + height, x : x + width])
                face = cv2.resize(face, FACE_SIZE)
                if not cv2.imwrite(str(capture_dir / f"{saved:03}.png"), face):
                    raise RuntimeError("Could not save a face sample.")
                saved += 1
                cv2.putText(
                    display,
                    f"Samples: {saved}/{sample_count}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (50, 210, 80),
                    2,
                )
            else:
                message = "Show one face" if not len(faces) else "Only one person at a time"
                cv2.putText(
                    display,
                    message,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (30, 30, 230),
                    2,
                )

            cv2.imshow("Register face - press Q to cancel", display)
            if cv2.waitKey(120) & 0xFF == ord("q"):
                break
    except Exception:
        shutil.rmtree(capture_dir, ignore_errors=True)
        raise
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if saved != sample_count:
        shutil.rmtree(capture_dir)
        print("Registration cancelled; no samples were kept.")
        return

    if person_dir.exists():
        shutil.rmtree(person_dir)
    capture_dir.rename(person_dir)
    people[person_id] = name.strip()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PEOPLE_FILE.write_text(json.dumps(people, indent=2), encoding="utf-8")
    print(f"Registered {name.strip()} (ID {person_id}) with {saved} face samples.")


def train_model():
    people = load_people()
    images = []
    labels = []
    for person_id in people:
        person_dir = FACES_DIR / person_id
        for image_path in sorted(person_dir.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(cv2.resize(image, FACE_SIZE))
                labels.append(int(person_id))

    if not images:
        raise RuntimeError("No face samples found. Register at least one person first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(MODEL_FILE))
    print(f"Trained with {len(images)} face samples for {len(people)} people.")


def load_attendance_for_today():
    attended = set()
    if ATTENDANCE_FILE.exists():
        workbook = load_workbook(ATTENDANCE_FILE, read_only=True, data_only=True)
        sheet = workbook["Attendance"]
        for recorded_date, _, person_id, *_ in sheet.iter_rows(
            min_row=2, values_only=True
        ):
            if isinstance(recorded_date, datetime):
                recorded_date = recorded_date.date()
            if recorded_date == date.today():
                attended.add(str(person_id))
        workbook.close()
    return attended


def record_attendance(person_id, name, attended):
    today = date.today().isoformat()
    if person_id in attended:
        return False

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ATTENDANCE_FILE.exists():
        workbook = load_workbook(ATTENDANCE_FILE)
        sheet = workbook["Attendance"]
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Attendance"
        sheet.append(["Date", "Time", "ID", "Name"])
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        sheet.freeze_panes = "A2"
        sheet.column_dimensions["A"].width = 14
        sheet.column_dimensions["B"].width = 12
        sheet.column_dimensions["C"].width = 10
        sheet.column_dimensions["D"].width = 24

    now = datetime.now()
    sheet.append([now.date(), now.time().replace(microsecond=0), int(person_id), name])
    sheet.cell(sheet.max_row, 1).number_format = "yyyy-mm-dd"
    sheet.cell(sheet.max_row, 2).number_format = "hh:mm:ss"
    sheet.auto_filter.ref = sheet.dimensions
    workbook.save(ATTENDANCE_FILE)
    workbook.close()
    attended.add(person_id)
    return True


def mark_attendance(threshold, test_mode=False):
    if not MODEL_FILE.exists():
        raise RuntimeError("No trained model found. Run the train command first.")

    people = load_people()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))
    detector = get_detector()
    attended = set() if test_mode else load_attendance_for_today()
    candidate_id = None
    candidate_frames = 0
    register_unknown = False
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam. Check that it is connected and available.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            gray, faces = find_faces(frame, detector)
            label = "No face detected"
            color = (30, 30, 230)
            current_id = None
            unknown_detected = False

            if len(faces) == 1:
                x, y, width, height = faces[0]
                face = cv2.equalizeHist(gray[y : y + height, x : x + width])
                face = cv2.resize(face, FACE_SIZE)
                predicted_id, confidence = recognizer.predict(face)
                person_id = str(predicted_id)
                if confidence <= threshold and person_id in people:
                    current_id = person_id
                    label = f"{people[person_id]} ({confidence:.0f})"
                    color = (50, 210, 80)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                else:
                    unknown_detected = True
                    label = "Unknown"
                    if not test_mode:
                        label += " - Press R to add"
                    candidate_id = None
                    candidate_frames = 0
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

            if current_id:
                if current_id == candidate_id:
                    candidate_frames += 1
                else:
                    candidate_id = current_id
                    candidate_frames = 1
                if candidate_frames >= 5:
                    if test_mode:
                        if candidate_frames == 5:
                            print(f"Test match: {people[current_id]} (ID {current_id})")
                        label += " - Match confirmed (test)"
                    elif current_id in attended:
                        label += " - Already present today"
                    else:
                        label += " - Press Y to record / N to skip"
            elif len(faces) != 1:
                candidate_id = None
                candidate_frames = 0
                if len(faces) > 1:
                    label = "Only one face at a time"

            cv2.putText(
                frame,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.putText(
                frame,
                "Q=stop" if test_mode else "Y=record N=skip R=add unknown Q=stop",
                (12, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Face attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if not test_mode and unknown_detected and key in (ord("r"), ord("R")):
                register_unknown = True
                break
            if not test_mode and candidate_frames >= 5:
                if key in (ord("y"), ord("Y")):
                    if record_attendance(candidate_id, people[candidate_id], attended):
                        print(f"Marked present: {people[candidate_id]} (ID {candidate_id})")
                elif key in (ord("n"), ord("N")):
                    candidate_id = None
                    candidate_frames = 0
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if register_unknown:
        print("Add the unknown person. Keep them in front of the camera for sample capture.")
        person_id = input("New numeric ID: ").strip()
        name = input("Name: ").strip()
        previous_people = load_people()
        register_person(person_id, name, 30)
        if load_people() != previous_people:
            train_model()
            print("Registration complete. Restart attendance to use the updated model.")


def main():
    parser = argparse.ArgumentParser(description="Local face-recognition attendance using OpenCV.")
    commands = parser.add_subparsers(dest="command", required=True)

    register_parser = commands.add_parser("register", help="Capture face samples for a person.")
    register_parser.add_argument("--id", required=True, help="Unique numeric ID for the person.")
    register_parser.add_argument("--name", required=True, help="Person's name.")
    register_parser.add_argument("--samples", type=int, default=30, help="Number of face samples (default: 30).")
    register_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing person's samples after a successful recapture.",
    )

    commands.add_parser("train", help="Train the face recognizer from registered samples.")

    attendance_parser = commands.add_parser("attendance", help="Run the webcam attendance check.")
    attendance_parser.add_argument(
        "--threshold",
        type=float,
        default=65.0,
        help="Maximum LBPH distance to accept a match (default: 65).",
    )
    attendance_parser.add_argument(
        "--test",
        action="store_true",
        help="Preview recognition without writing attendance records.",
    )

    args = parser.parse_args()
    try:
        if args.command == "register":
            if args.samples < 1:
                parser.error("--samples must be at least 1")
            register_person(args.id, args.name, args.samples, replace=args.replace)
        elif args.command == "train":
            train_model()
        else:
            mark_attendance(args.threshold, test_mode=args.test)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()