#!/bin/bash
# Installs the agent's Python libraries from vendor/wheels, without internet access.
# (Cloud environments may block pypi.org.) Claude Code is already installed there.
set -e
cd "$(dirname "$0")"
pip install -q --no-index --find-links vendor/wheels alpaca-py==0.44.0 pandas==3.0.6 \
  || pip install -q --break-system-packages --no-index --find-links vendor/wheels alpaca-py==0.44.0 pandas==3.0.6
python -c "import alpaca, pandas; print('dependencies ok')"
