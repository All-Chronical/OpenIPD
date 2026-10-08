# PyInstaller spec for platform-native one-folder executable bundles.
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [("face_landmarker.task", ".")]
binaries = []
hiddenimports = []

datas += collect_data_files("mediapipe", excludes=["**/test/**"])
datas += collect_data_files("cv2")
binaries += collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("cv2")
hiddenimports += ["mediapipe.tasks.python.vision"]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OpenIPD",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OpenIPD",
)