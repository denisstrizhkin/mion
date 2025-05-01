#!/usr/bin/env fish

for in in ./*.png
    set out (path change-extension .webp $in)
    magick $in -quality 85 $out
end
