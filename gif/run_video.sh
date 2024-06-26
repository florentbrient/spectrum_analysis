cas='FIRE2Dreal' # 'ASTEX2D'  'FIRE2Dreal' #'BOMEX2D' #'IHOP2' #'FIR2D' #'IHOP0' #'FIRE' #IHOP
EXP='FIR2D' #'F2DNW' #'B2DNW' #'IHOP2' #'FIR2D' #'IHOP0' #RH04
typs=('Mean') #(Mean Anom)
vars=('SVT004' 'SVT006')
model='V5-5-1'
joingraph=0


for var in "${vars[@]}";do
   varch+="$var"
done
echo $varch

version='V0001' #'V0302_00' #'00' #V0302 #0_0 #default

start=001
delay=10
loop=0

path='../figures/'$model'/2D/'$cas'/'$EXP'/'

# ../figures/2D/IHOP2DNW/IHOP2/SVT006SVT004_join
# SVT004SVT006_join_IHOP2_V0001_720.png

for tt in "${typs[@]}"
do

ch=''
if [ $tt == 'Anom' ]
then
ch=$ch'_anom'
fi

#for var in "${varch[@]}"
#do

if [ $joingraph == 1 ]
then
ch=$ch'_join'
fi

var=$varch$ch
path=$path$var'/'

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



