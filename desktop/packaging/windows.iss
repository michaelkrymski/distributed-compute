[Setup]
AppName=Distributed Compute
AppVersion=0.2.0
DefaultDirName={localappdata}\Programs\Distributed-Compute
DefaultGroupName=Distributed Compute
PrivilegesRequired=lowest
OutputDir=installer-output
OutputBaseFilename=Distributed-Compute-Setup
Compression=lzma2
SolidCompression=yes
[Files]
Source: "{#SourceDir}\dist\Distributed-Compute\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\Distributed Compute"; Filename: "{app}\Distributed-Compute.exe"
[Run]
Filename: "{app}\Distributed-Compute.exe"; Description: "Enroll this computer securely and complete setup"; Flags: postinstall nowait skipifsilent
