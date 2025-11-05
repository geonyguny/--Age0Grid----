import argparse, shutil, os, glob
p=argparse.ArgumentParser(); p.add_argument("--src"); p.add_argument("--dst"); a=p.parse_args()
os.makedirs(a.dst,exist_ok=True)
n=0
for f in glob.glob(os.path.join(a.src,"*.png"))+glob.glob(os.path.join(a.src,"*pivot.csv")):
    shutil.copy2(f, os.path.join(a.dst, os.path.basename(f))); n+=1
print(f"[OK] publish {n} files -> {a.dst}")
