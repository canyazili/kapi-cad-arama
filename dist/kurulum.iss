; Kapı CAD Arama — tek dosyalık kurulum paketi
; Derleme: ISCC.exe kurulum.iss  (kaynak: dist\paket, çıktı: Masaüstü)
[Setup]
AppName=Kapı CAD Arama
AppVersion=1.0
AppPublisher=canya
DefaultDirName={localappdata}\KapiArama
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
DisableWelcomePage=no
Compression=lzma2/fast
SolidCompression=no
OutputDir=C:\Users\canya\Desktop
OutputBaseFilename=KapiArama_Kurulum
WizardStyle=modern
SetupIconFile=
UninstallDisplayName=Kapı CAD Arama

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Files]
Source: "c:\Users\canya\Desktop\kapı\dist\paket\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userdesktop}\Kapı CAD Arama"; Filename: "{app}\KapiArama.exe"; WorkingDir: "{app}"
Name: "{userprograms}\Kapı CAD Arama"; Filename: "{app}\KapiArama.exe"; WorkingDir: "{app}"

[Run]
Filename: "{app}\KapiArama.exe"; Description: "Uygulamayı şimdi başlat"; Flags: postinstall nowait skipifsilent
