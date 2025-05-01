#!/usr/bin/env fish

for img in ./*.webp
    echo $img
    magick $img -resize '90x50!' $img
end
