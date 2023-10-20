cas='FIRE' #IHOP
EXP='F2DNW' #RH04
typs=('Mean') #(Mean Anom)
variables=(SVT006) #(RCT DTHRAD CLDFR THV THT RNPM THLM SVT001 SVT002 SVT003 SVT004 SVT005 SVT006 UT VT WT PABST)
#var1D=(LWP LFC LCL)

version='V0001' #'V0302_00' #'00' #V0302 #0_0 #default

start=001
delay=10
loop=0

path='../figures/'

# ../figures/SVT004_F2DNW_V0001_720.png

#for tt in "${typs[@]}"
#do

for var in "${variables[@]}"
do

echo $var
namefile=${path}'TESTGIF/'$var'_'$EXP'_'$version'_%03d.png'
nameanimated=${path}'Animated_'$var'_'$EXP'_'$version'.mp4'
namegif=$path'GIF_'$var'_'$EXP'_'$version'.gif'

#for hour in "$hours[@]}"
#do
echo '****** FILE IN  :  '$namefile
echo '****** FILE OUT :  '$nameanimated
echo ''
echo '***** MAKE AVI *******'
#ffmpeg  -f image2 -framerate 1 -pattern_type sequence -start_number $start -i $path/${tt}_$var$char1$char2$cross$char3$version%d.png  -crf 50 -y -vcodec mpeg4 $path/Animated_${tt}_${var}_${cross}$versch.avi
#ffmpeg  -f image2 -framerate 1 -pattern_type glob -start_number $start -i $namefile  -crf 50 -y -vcodec mpeg4 $nameanimated

ffmpeg  -framerate 30 -i $namefile -crf 10 -level 4.2 -preset slow -vcodec libx264 $nameanimated


echo '***** MAKE GIF *******'
#convert -delay $delay -loop $loop $namefile $namegif


#done
done



