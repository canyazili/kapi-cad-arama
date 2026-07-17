; FUAR AHŞAP - Kapı Arama — Inno Setup kurulum betiği
; Kişiye özel (admin gerektirmez) kurulum; masaüstü + Başlat menüsü kısayolu.
; Derleme: ISCC.exe kurulum.iss  -> FUARAHSAP_Kurulum.exe (proje kökünde)
; NOT: Bu dosya UTF-8 BOM ile kaydedilmeli (Türkçe karakterler için).

#define AppName "FUAR AHSAP - Kapi Arama"
#define AppNameTR "FUAR AHŞAP — Kapı Arama"
#define AppShort "FUAR AHŞAP"
#define AppVersion "1.0"
#define AppPublisher "Muratcan Yazılı"
#define AppExe "KapiArama.exe"

[Setup]
AppId={{4C327AE9-2DD3-4B2A-8C1E-E16D65F961B7}
AppName={#AppNameTR}
AppVersion={#AppVersion}
AppVerName={#AppNameTR} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\FUAR AHSAP
DefaultGroupName={#AppShort}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=.
OutputBaseFilename=FUARAHSAP_Kurulum
SetupIconFile=marka\app_icon.ico
Compression=lzma2/normal
SolidCompression=no
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppNameTR}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "tr"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon"; Description: "Masaüstünde simge oluştur"; GroupDescription: "Ek kısayollar:"; Flags: checkedonce

[Files]
; --- Uygulama dosyaları: her güncellemede yenilenir ---
Source: "dist\KapiArama\KapiArama.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\KapiArama\_internal\*"; DestDir: "{app}\_internal"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "dist\KapiArama\configs\*"; DestDir: "{app}\configs"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "dist\KapiArama\marka\*"; DestDir: "{app}\marka"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "dist\KapiArama\modeller\*"; DestDir: "{app}\modeller"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "dist\KapiArama\araclar\*"; DestDir: "{app}\araclar"; Flags: recursesubdirs createallsubdirs ignoreversion
; --- Müşteri verisi (eklenen kapı/çizim/indeks): YALNIZ ilk kurulumda konur;
;     güncellemede ve program kaldırmada KORUNUR (müşterinin ekledikleri silinmesin) ---
Source: "dist\KapiArama\data\*"; DestDir: "{app}\data"; Flags: recursesubdirs createallsubdirs onlyifdoesntexist uninsneveruninstall
Source: "dist\KapiArama\index\*"; DestDir: "{app}\index"; Flags: recursesubdirs createallsubdirs onlyifdoesntexist uninsneveruninstall

[Icons]
Name: "{group}\{#AppShort}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppShort}}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppShort}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppShort}}"; Flags: nowait postinstall skipifsilent
