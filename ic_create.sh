#!/bin/bash
# Usage :: ic_create.sh <param_file>

if [ $# -ne 1 ]; then
    echo 'Usage : ic_create.sh <param_file>'
    exit 1
fi

if [ ! -e $1 ]; then
    echo 'file "'$1'" not found'
    exit 1
fi


grafic_file=`basename $1 .inp`.grafic

#python3 make_class_transfer.py $1 $grafic_file
#./graficlass $grafic_file > grafic1.log


