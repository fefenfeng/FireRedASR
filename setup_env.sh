#!/bin/bash
# setup_env.sh - 项目环境设置脚本

export PATH="$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH"
export PYTHONPATH="$PWD/:$PYTHONPATH"
echo "环境变量已设置 for $(basename $PWD)"