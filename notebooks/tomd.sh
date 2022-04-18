#!/bin/sh
# Please run it at `notebook` directory
filepath=$(basename "$1")  # filename
parentpath=$(dirname `pwd`)  # dirname
 
output=$(echo "${parentpath}/book/${filepath%%.*}.md")  # remove the extension and add new extension
jupytext $1 --to myst --output ${output} 