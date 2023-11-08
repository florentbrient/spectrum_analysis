cas='BOMEX2D' #'IHOP2' #'FIR2D' #'IHOP0' #'FIRE' #IHOP
EXP='B2DNW' #'IHOP2' #'FIR2D' #'IHOP0' #RH04
typs=('Mean') #(Mean Anom)
variables=(SVT004WT)
joingraph=0

version='V0001' #'V0302_00' #'00' #V0302 #0_0 #default

start=001
delay=10
loop=0

path='../figures/'$cas'/'$EXP'/'

# ../figures/SVT004_F2DNW_V0001_720.png

#for tt in "${typs[@]}"
#do

for var in "${variables[@]}"
do

if [ $joingraph == 1 ]
then
var=$var'_join'
fi


echo $var
namefile=${path}$var'_'$EXP'_'$version'_%03d.png'
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

ffmpeg  -framerate 10 -i $namefile -crf 10 -level 4.2 -preset slow -vcodec libx264 $nameanimated


echo '***** MAKE GIF *******'
#convert -delay $delay -loop $loop $namefile $namegif


#done
done



