# AttendX | Face Recognition Attendance

A local webcam attendance tool using OpenCV face detection and LBPH recognition. Face images, the trained model, and the Excel attendance workbook are stored in the `attendance_data` folder beside the script.

## Setup

Use Python 3.10 or newer. From this folder, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Use

Launch the desktop interface for camera monitoring, test mode, enrollment, training, and Excel records:

```powershell
python attendance_gui.py
```

Register each person with a unique numeric ID. The webcam window collects samples while exactly one face is visible; press `Q` to cancel.

```powershell
python attendance.py register --id 1 --name "Alex Smith"
python attendance.py register --id 2 --name "Jamie Lee"
python attendance.py train
python attendance.py attendance --test
python attendance.py attendance
```

To correct a person's face samples, recapture that person with the same ID and `--replace`, then retrain. The old samples are kept if the recapture is cancelled.

```powershell
python attendance.py register --id 1 --name "Alex Smith" --replace
python attendance.py train
```

Use `attendance --test` to preview recognition without writing attendance. In normal attendance mode, press `Y` to confirm the displayed person or `N` to skip them. When one face is shown as Unknown, press `R` to add them; enter a new numeric ID and name in the terminal, then capture their samples. The model retrains automatically, and you must restart attendance to use it. Attendance is saved to `attendance_data/attendance.xlsx`, with at most one entry per person per day. Open the workbook in Excel to view or filter records. Press `Q` in the webcam window to stop. The recognition threshold can be adjusted with `--threshold`; lower values are stricter, and the right value depends on lighting and camera conditions.

## Privacy and limitations

Get informed permission before collecting or using face images. The samples and model are stored locally without encryption; restrict access to the project folder and delete `attendance_data` when the data is no longer needed. This basic LBPH system can make false matches and is not liveness detection, so review attendance records and do not use it as the sole basis for consequential decisions.