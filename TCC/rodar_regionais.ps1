<#
  rodar_regionais.ps1 - roda as regionais do TCC EM SEQUENCIA (uma por vez).

  - Espera o download que ja esta rodando terminar (detecta pelo log parado).
  - Para cada regional roda ate 3 passadas; para antes se a passada
    terminar com err=0 (o resume faz as repetidas serem rapidas).
  - Loga em D:\CityVision\TCC-data\log_<acervo>_<regiao>_z<zoom>.txt
    (-Append, nao apaga o anterior; nome inclui o acervo para nao misturar
    logs de epocas diferentes, ex.: 2019 e 2012).

  Uso:
    powershell -ExecutionPolicy Bypass -File D:\CityVision\TCC\rodar_regionais.ps1
    ... -Acervo Ortofotos2012      (baixa outra epoca; tiles ficam em
                                    pasta separada, log tambem)
    ... -SemEspera            (nao espera nada, comeca ja)
    ... -Regioes matriz,cic   (roda so essas, na ordem dada)
    ... -DryRun               (so conta os tiles, nao baixa)
#>
param(
  [string[]]$Regioes = @('matriz','boa_vista','santa_felicidade','portao',
                         'boqueirao','cajuru','cic','pinheirinho'),
  [switch]$SemEspera,
  [switch]$DryRun,
  [int]$Threads = 16,
  [int]$MaxPassadas = 3,
  [string]$Saida = 'D:\CityVision\TCC-data',
  [string]$Acervo = 'Ortofotos2019',
  [int]$Zoom = 20
)

# com -File, "-Regioes a,b" chega como UMA string; separa na virgula.
$Regioes = $Regioes -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }

$ErrorActionPreference = 'Continue'
$script = 'D:\CityVision\TCC\baixar_ortofotos_ippuc.py'
if (-not (Test-Path $script)) { throw "Nao achei $script" }
if (-not (Test-Path $Saida))  { New-Item -ItemType Directory -Force $Saida | Out-Null }

function Write-Cab($txt) {
  Write-Host ''
  Write-Host ('=' * 64) -ForegroundColor Cyan
  Write-Host $txt -ForegroundColor Cyan
  Write-Host ('=' * 64) -ForegroundColor Cyan
}

# ---------------------------------------------------------------- espera
# Considera que ainda ha download rodando enquanto QUALQUER log_*.txt
# do D:\TCC-data for modificado nos ultimos $ociosoSeg segundos.
function Wait-DownloadAtual {
  $ociosoSeg = 180
  while ($true) {
    $logs = Get-ChildItem (Join-Path $Saida 'log_*.txt') -ErrorAction SilentlyContinue
    if (-not $logs) { return }
    $ultimo = ($logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
    $idle = (New-TimeSpan -Start $ultimo.LastWriteTime -End (Get-Date)).TotalSeconds
    if ($idle -ge $ociosoSeg) {
      Write-Host ("[espera] nenhum log mexeu ha {0:N0}s - seguindo." -f $idle) -ForegroundColor Yellow
      return
    }
    Write-Host ("[espera] {0} ainda ativo ({1:N0}s atras). Checo de novo em 60s..." -f `
                $ultimo.Name, $idle) -ForegroundColor DarkGray
    Start-Sleep -Seconds 60
  }
}

# ---------------------------------------------------------------- passada
function Invoke-Regional($nome) {
  $log = Join-Path $Saida "log_${Acervo}_${nome}_z$Zoom.txt"
  for ($p = 1; $p -le $MaxPassadas; $p++) {
    Write-Cab ("[{0}] {1} - passada {2}/{3}" -f (Get-Date -Format 'HH:mm:ss'), $nome, $p, $MaxPassadas)
    $cmdArgs = @('--acervo', $Acervo, '--regiao', "regional_$nome",
              '--zoom', $Zoom, '--saida', $Saida, '--threads', $Threads)
    if ($DryRun) { $cmdArgs += '--dry-run' }

    "`n===== passada $p - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" |
      Out-File -FilePath $log -Append -Encoding utf8

    # ErrorRecord -> string antes de gravar: evita o ruido "NativeCommandError"
    # e mantem o log todo em UTF-8 (Tee-Object grava UTF-16 e mistura).
    # ErrorRecord -> string e gravacao linha a linha (StreamWriter com AutoFlush):
    # evita o ruido "NativeCommandError", mantem o log em UTF-8 e faz o arquivo
    # acompanhar o download ao vivo (Out-File/Tee-Object bufferizam ate o fim).
    $sw = New-Object System.IO.StreamWriter($log, $true, [System.Text.UTF8Encoding]::new($false))
    $sw.AutoFlush = $true
    try {
      & python $script @cmdArgs 2>&1 | ForEach-Object { $linha = "$_"; Write-Host $linha; $sw.WriteLine($linha) }
    } finally { $sw.Close() }

    if ($DryRun) { return }

    # Le a ultima linha "Concluido ... err=N" pra decidir se repete.
    $fim = Select-String -Path $log -Pattern 'err=(\d+)\s*$' -ErrorAction SilentlyContinue |
           Select-Object -Last 1
    if ($fim -and [int]$fim.Matches[0].Groups[1].Value -eq 0) {
      Write-Host "[$nome] passada $p terminou com err=0 - proxima regional." -ForegroundColor Green
      return
    }
    if ($fim) {
      Write-Host ("[{0}] ainda com {1} erros - repetindo (resume pula o que ja tem)." -f `
                  $nome, $fim.Matches[0].Groups[1].Value) -ForegroundColor Yellow
    } else {
      Write-Host "[$nome] passada $p nao chegou ao fim (queda?) - repetindo." -ForegroundColor Yellow
    }
    Start-Sleep -Seconds 20
  }
  Write-Host "[$nome] fim das $MaxPassadas passadas - seguindo mesmo assim." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- main
$t0 = Get-Date
Write-Cab "FILA: $($Regioes -join ' -> ')"
if (-not $SemEspera) { Wait-DownloadAtual }

foreach ($r in $Regioes) { Invoke-Regional $r }

Write-Cab ("TUDO FEITO em {0:N1} h" -f ((Get-Date) - $t0).TotalHours)
Write-Host 'Contagem de tiles no disco:'
$dir = Join-Path $Saida "$Acervo\$Zoom"
if (Test-Path $dir) {
  $n = (Get-ChildItem $dir -Recurse -Filter *.jpg -File | Measure-Object).Count
  Write-Host ("  {0:N0} arquivos .jpg em {1}" -f $n, $dir)
}
