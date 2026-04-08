Hub and casing curve files must have leading spaces removed from each line,
and also must have multiple spaces replaced with single spaces.

This can be done as follows:

sed -i 's/^[[:blank:]]*//' <filename>
tr -s ' ' < <filename> > <updated_filename>

where <filename> is the original hub or casing curve file, and
<updated_filename> is the new name for the file you actually
want to use.
