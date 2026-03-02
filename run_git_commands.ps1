cd 'E:\Code\Rust\wt-a770-out-head'

Write-Host "=== Running: git add -A ===" -ForegroundColor Green
git add -A
Write-Host "git add completed`n" -ForegroundColor Green

Write-Host "=== Running: git commit ===" -ForegroundColor Green
$commitMessage = @"
feat(kernels): add opencl_output_head for A770 vocabulary projection

- OutputHead with configurable hidden/vocab dimensions
- TiedWeights support (shared embedding matrix)
- PartialVocabDecoder for efficient top-K
- LogitNormalizer for numerical stability
- EfficientProjection with tiled matmul
- 50 tests including property tests

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
"@
git commit -m $commitMessage
Write-Host "git commit completed`n" -ForegroundColor Green

Write-Host "=== Running: git push -u origin opencl/output-projection-head ===" -ForegroundColor Green
git push -u origin opencl/output-projection-head
Write-Host "git push completed" -ForegroundColor Green
