# In your Aurora folder (D:\Aurora)
cat > build.sh << 'EOF'
#!/bin/bash
set -euo pipefail
echo "=== Building model artifacts ==="
python -u model.py 2>&1 | tee build.log
echo "=== Build complete ==="
ls -la artifacts/
EOF