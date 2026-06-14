; Inno Setup script for Dupless Video — wraps the PyInstaller COLLECT output into a
; double-click installer for a NON-TECHNICAL user (no Python, no PATH, no admin needed).
;
; Build order (see docs/BUILD_WINDOWS.md):
;   pyinstaller --noconfirm dupless-video.spec     ->  dist\Dupless Video\
;   iscc install\windows\dupless-video.iss          ->  Output\DuplessVideoSetup.exe
;
; Per-user install (PrivilegesRequired=lowest): lands in %LOCALAPPDATA%\Programs, so the friend
; never sees a UAC prompt. The app's DATA (DB, embeddings) lives separately in
; %LOCALAPPDATA%\Dupless Video (runtime.app_data_dir); uninstall KEEPS it unless the user opts to
; remove it (prompt). Optional: "start with Windows" task writes the same HKCU\Run value the
; in-app toggle uses (startup.py), so they never conflict.

#define AppName "Dupless Video"
#define AppVersion "0.1.0"
#define AppPublisher "Dupless Video"
#define AppExeName "Dupless Video.exe"

; Optional build flavor (CPU / GPU) -> distinct installer filenames so both can coexist as downloads.
; SAME AppId/AppName on purpose: it is the one app (only the bundled torch differs), so installing one
; cleanly upgrades the other in place. Pass with:  iscc /DFlavor=CPU  (or /DFlavor=GPU). Default: unnamed.
#ifndef Flavor
  #define Flavor ""
#endif
#if Flavor != ""
  #define OutSuffix "-" + Flavor
#else
  #define OutSuffix ""
#endif

[Setup]
AppId={{B3D7F2A1-5C4E-4A9B-8E2D-DUPLESSVIDEO01}}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename=DuplessVideoSetup{#OutSuffix}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; If the app is RUNNING, setup AND the uninstaller detect this mutex (created by app_entry
; _hold_app_mutex) and ask the user to close it first — never replace/remove files in use.
AppMutex=DuplessVideoSetupMutex
; Also use the Restart Manager to auto-offer closing any app holding install files open.
CloseApplications=yes
RestartApplications=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\..\src\dupdetect\ui\icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
; Shown on a page right before Finish: the BETA "review before deleting" note.
InfoAfterFile=beta-note.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
; (1) Optional "start with Windows" — OFF by default; writes the HKCU\Run value below.
Name: "startupwithwindows"; Description: "{cm:StartupTask}"; GroupDescription: "{cm:StartupGroup}"; Flags: unchecked

[Files]
; the entire PyInstaller COLLECT folder (exe + Python + torch + Qt + the bundle: model & ffmpeg)
Source: "..\..\dist\Dupless Video\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; (1) Run-at-login. ValueName matches startup.py (_APP_NAME="Dupless Video") so the in-app toggle and
; this installer task share ONE entry. uninsdeletevalue removes it on uninstall.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; \
    ValueName: "{#AppName}"; ValueData: """{app}\{#AppExeName}"" --tray"; Flags: uninsdeletevalue; \
    Tasks: startupwithwindows

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[CustomMessages]
english.StartupTask=Start {#AppName} automatically when Windows starts
english.StartupGroup=Startup:
english.RemoveDataPrompt=Also delete your {#AppName} database and cache?%n(%1)%n%nYes = remove everything.  No = keep your index so a reinstall finds your library.

[Code]
{ (3) On uninstall, OFFER to remove the per-user data (DB + embeddings). Default keeps it: the index
  is expensive to rebuild and a reinstall should find the library. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\{#AppName}');
    if DirExists(DataDir) then
      if MsgBox(FmtMessage(CustomMessage('RemoveDataPrompt'), [DataDir]), mbConfirmation, MB_YESNO) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
