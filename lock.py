import base64
import zlib
import glob
import os

# Find all python files in the engine folder
for filepath in glob.glob("engine/*.py"):
    if "__init__" in filepath: 
        continue
        
    with open(filepath, 'rb') as f:
        code = f.read()
    
    # Skip if already encrypted
    if b"base64.b64decode" in code: 
        continue
    
    # Compress and encode the source code into unreadable text
    encoded = base64.b64encode(zlib.compress(code)).decode('utf-8')
    
    # Overwrite the file with the execution wrapper
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"import base64, zlib\nexec(zlib.decompress(base64.b64decode('{encoded}')), globals())\n")
        
print("✅ AI Engine successfully encrypted and locked!")