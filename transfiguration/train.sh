echo "Training for $1 epoch(es)."

py ./style.py $1

echo "Training completed. Reboot in 30s."

shutdown -r -t 30