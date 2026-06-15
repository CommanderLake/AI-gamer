param(
	[string]$Directory = "L:\TrainingData"
)

$source = @'
using System;
using System.Collections.Generic;
using System.IO;

public sealed class CausalRepairResult
{
	public string Name;
	public long OriginalBytes;
	public long CorrectedBytes;
	public long CompleteRecords;
	public long CorrectedRecords;
	public long DiscardedTailBytes;
}

public static class CausalTrainingDataRepair
{
	private const int HeaderBytes = 8;
	private const int InputStateBytes = 12;
	private const int StreamBufferBytes = 4 * 1024 * 1024;

	private static void ReadExactly(Stream stream, byte[] buffer, int count)
	{
		int offset = 0;
		while (offset < count)
		{
			int read = stream.Read(buffer, offset, count - offset);
			if (read == 0)
				throw new EndOfStreamException("Unexpected end of training data.");
			offset += read;
		}
	}

	private static void VerifyRange(FileStream original, FileStream corrected, long originalOffset, long correctedOffset, int count)
	{
		byte[] originalData = new byte[count];
		byte[] correctedData = new byte[count];
		original.Position = originalOffset;
		corrected.Position = correctedOffset;
		ReadExactly(original, originalData, count);
		ReadExactly(corrected, correctedData, count);
		for (int i = 0; i < count; ++i)
		{
			if (originalData[i] != correctedData[i])
				throw new InvalidDataException("Corrected data verification failed.");
		}
	}

	private static void Verify(string originalPath, string correctedPath, int frameBytes, long correctedRecords)
	{
		long recordBytes = InputStateBytes + (long)frameBytes;
		long[] samples = { 0, correctedRecords / 2, correctedRecords - 1 };
		using (FileStream original = new FileStream(originalPath, FileMode.Open, FileAccess.Read, FileShare.Read, StreamBufferBytes, FileOptions.RandomAccess))
		using (FileStream corrected = new FileStream(correctedPath, FileMode.Open, FileAccess.Read, FileShare.Read, StreamBufferBytes, FileOptions.RandomAccess))
		{
			VerifyRange(original, corrected, 0, 0, HeaderBytes);
			foreach (long sample in samples)
			{
				long correctedRecordOffset = HeaderBytes + sample * recordBytes;
				long nextOriginalRecordOffset = HeaderBytes + (sample + 1) * recordBytes;
				long originalFrameOffset = HeaderBytes + sample * recordBytes + InputStateBytes;
				VerifyRange(original, corrected, nextOriginalRecordOffset, correctedRecordOffset, InputStateBytes);
				VerifyRange(original, corrected, originalFrameOffset, correctedRecordOffset + InputStateBytes, frameBytes);
			}
		}
	}

	public static CausalRepairResult Repair(string path)
	{
		string tempPath = path + ".causal.tmp";
		string backupPath = path + ".precausal.bak";
		string cachePath = path + ".idxcache";
		string cacheBackupPath = cachePath + ".precausal.bak";
		if (File.Exists(tempPath) || File.Exists(backupPath))
			throw new IOException("Temporary or backup file already exists for " + path);
		if (File.Exists(cachePath) && File.Exists(cacheBackupPath))
			throw new IOException("Index cache backup already exists for " + path);

		FileInfo originalInfo = new FileInfo(path);
		long originalBytes = originalInfo.Length;
		DateTime originalWriteTimeUtc = originalInfo.LastWriteTimeUtc;
		int width;
		int height;
		int frameBytes;
		long recordBytes;
		long completeRecords;
		long discardedTailBytes;
		long correctedRecords;

		try
		{
			using (FileStream input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, StreamBufferBytes, FileOptions.SequentialScan))
			using (FileStream output = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, StreamBufferBytes, FileOptions.SequentialScan))
			{
				byte[] header = new byte[HeaderBytes];
				ReadExactly(input, header, HeaderBytes);
				width = BitConverter.ToInt32(header, 0);
				height = BitConverter.ToInt32(header, 4);
				long frameBytesLong = (long)width * height * 3;
				if (width <= 0 || height <= 0 || frameBytesLong > Int32.MaxValue)
					throw new InvalidDataException("Invalid frame dimensions in " + path);
				frameBytes = (int)frameBytesLong;
				recordBytes = InputStateBytes + frameBytesLong;
				long payloadBytes = originalBytes - HeaderBytes;
				completeRecords = payloadBytes / recordBytes;
				discardedTailBytes = payloadBytes % recordBytes;
				if (completeRecords < 2)
					throw new InvalidDataException("At least two complete records are required in " + path);
				correctedRecords = completeRecords - 1;

				byte[] ignoredFirstInput = new byte[InputStateBytes];
				byte[] nextInput = new byte[InputStateBytes];
				byte[] frame = new byte[frameBytes];
				output.Write(header, 0, header.Length);
				ReadExactly(input, ignoredFirstInput, ignoredFirstInput.Length);

				long progressInterval = Math.Max(1, correctedRecords / 100);
				for (long record = 0; record < correctedRecords; ++record)
				{
					ReadExactly(input, frame, frame.Length);
					ReadExactly(input, nextInput, nextInput.Length);
					output.Write(nextInput, 0, nextInput.Length);
					output.Write(frame, 0, frame.Length);
					if ((record + 1) % progressInterval == 0 || record + 1 == correctedRecords)
					{
						int percent = (int)((record + 1) * 100 / correctedRecords);
						Console.WriteLine("{0}: {1}% ({2:N0}/{3:N0})", Path.GetFileName(path), percent, record + 1, correctedRecords);
					}
				}
				output.Flush(true);
			}

			long expectedBytes = HeaderBytes + correctedRecords * recordBytes;
			if (new FileInfo(tempPath).Length != expectedBytes)
				throw new InvalidDataException("Corrected file has an unexpected size: " + tempPath);
			Verify(path, tempPath, frameBytes, correctedRecords);

			File.Move(path, backupPath);
			try
			{
				File.Move(tempPath, path);
			}
			catch
			{
				File.Move(backupPath, path);
				throw;
			}
			File.SetLastWriteTimeUtc(path, originalWriteTimeUtc);
			if (File.Exists(cachePath))
				File.Move(cachePath, cacheBackupPath);
		}
		catch
		{
			if (File.Exists(tempPath))
				File.Delete(tempPath);
			throw;
		}

		return new CausalRepairResult
		{
			Name = Path.GetFileName(path),
			OriginalBytes = originalBytes,
			CorrectedBytes = new FileInfo(path).Length,
			CompleteRecords = completeRecords,
			CorrectedRecords = correctedRecords,
			DiscardedTailBytes = discardedTailBytes
		};
	}
}
'@

Add-Type -TypeDefinition $source -Language CSharp

$files = Get-ChildItem -LiteralPath $Directory -Filter "*.bin" -File | Sort-Object Name
if ($files.Count -eq 0) {
	throw "No .bin training data files were found in $Directory"
}

$results = foreach ($file in $files) {
	Write-Host "Correcting $($file.FullName)"
	[CausalTrainingDataRepair]::Repair($file.FullName)
}

$manifestPath = Join-Path $Directory "causal-correction-manifest.txt"
$manifest = @(
	"AI Gamer causal training-data correction"
	"Completed UTC: $([DateTime]::UtcNow.ToString('O'))"
	"Transformation: output frame i is paired with original input state i+1."
	"The final complete frame and any incomplete trailing bytes were discarded."
	"Original files and index caches are retained with the .precausal.bak suffix."
	""
)
$manifest += $results | ForEach-Object {
	"{0}: originalBytes={1}; correctedBytes={2}; completeRecords={3}; correctedRecords={4}; discardedTailBytes={5}" -f
		$_.Name, $_.OriginalBytes, $_.CorrectedBytes, $_.CompleteRecords, $_.CorrectedRecords, $_.DiscardedTailBytes
}
$manifest | Set-Content -LiteralPath $manifestPath -Encoding ASCII
$results | Format-Table Name, CompleteRecords, CorrectedRecords, DiscardedTailBytes, OriginalBytes, CorrectedBytes -AutoSize
