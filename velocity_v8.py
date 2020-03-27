
#animation

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from scipy import stats
from celluloid import Camera

def nnp(mpp, ball_dia):
	rad_pix=0.5*ball_dia/mpp
	return math.pi*rad_pix**2


def ball_check(flow_spot, ball):
	if abs(flow_spot-ball) <= .2*ball:
		return True
	else:
		return False

def recenter(data, mindata):
	return data-mindata

#set some experiment dependent constants

mpp=2.307 #for Tom's fast camera
ball_dia=49 #in microns
fps=5

#nominal ball area
ball_area=nnp(2.307, 49)

#open file
vidObj=cv2.VideoCapture('20190716/dodecanol_45deg_03_RT.avi')

#read first frame
ret, frame1=vidObj.read()

#convert to grayscale
prvs=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

#set frame count
count=1
count2=0

cX_prvs=[]
cY_prvs=[]
time=[]
disp_microns=[0]


fig, (ax, ax2)=plt.subplots(ncols=2)
fig.set_size_inches(9.5,4.5)
fig.subplots_adjust(wspace=.5)
ax.set_xticks([])
ax.set_yticks([])
ax2.set_xlim([-.1, 10])
ax2.set_ylim([-.1, 250])
ax2.set_xlabel('time, s')
ax2.set_ylabel('displacement, microns')

camera = Camera(fig)
plt.ion()

#advance through video
while True:
	ret,frame2=vidObj.read()

	count+=1
#	print(count)

	if ret == True:
		next=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
	
		#calculate optical flow
		flow=cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3,15,3,5,1.2,0)

		#convert from dx,dy to mag and ang
		mag, ang=cv2.cartToPolar(flow[...,0], flow[...,1])
     
		#threshold second image
		ret2,th_next = cv2.threshold(next, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#invert threshold
		th_next_inv=cv2.bitwise_not(th_next)
		
		#convert mag of optical flow disp to right datatype
		T=mag.flatten()
		mag1=255*mag.astype(np.float32)/max(T)
		mag_img_u=mag1.astype(np.uint8)

		#threshold the magnitude (should be one large ball
		ret3, th_mag = cv2.threshold(mag_img_u, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		#stick these together (should be isolated area comparable to ball area) 
		comb_im= cv2.bitwise_and(th_next_inv,th_mag)
	
		#calculate size in pixels
		bins2 = np.linspace(0, 255, 256)
		HT2=np.histogram(comb_im, bins=bins2)
		movingspot_area=int(HT2[0][-1])

		prvs=next
		prvs2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

		#if data passes this quality check
		if ball_check(movingspot_area, ball_area)==True:
			
			#calculate ball "center" using moments
			M = cv2.moments(comb_im)
	
			#coordinates
			cX = M["m10"] / M["m00"]
			cY = M["m01"] / M["m00"]

			cX_prvs.append(cX)
			cY_prvs.append(cY)
			
			count2+=1

			if count2>1:		
				#displacement
				disp_mag=math.sqrt((cX_prvs[-1]-cX_prvs[-2])**2+(cY_prvs[-1]-cY_prvs[-2])**2)
				disp_microns_t=(disp_mag*mpp)+disp_microns[-1]
				disp_microns.append(disp_microns_t)
				time.append(count/fps)

				disp_microns_a=np.array(disp_microns[1:])
				time_a=np.array(time)

				disp_microns_ar=recenter(disp_microns_a, min(disp_microns_a))
				time_ar=recenter(time_a, min(time_a))
				
				cv2.circle(prvs2, (int(cX), int(cY)), 4, (0, 255, 0), -1)

				ax.imshow(prvs2)
				ax2.title.set_text('')
#				ax2.plot(count/fps-min(time), disp_microns_t, 'go', ms=2)
				ax2.plot(time_ar, disp_microns_ar, 'go', ms=2)
				asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
				asp /= np.abs(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
				ax2.set_aspect(asp)
				plt.pause(0.001)

				camera.snap()

	else:
		break



#plt.xlabel('time, seconds')
#plt.ylabel('position, microns')

#slope, intercept, r_value, p_value, std_err = stats.linregress(time_ar,disp_microns_ar)
#line = slope*time_ar+intercept
#ax2.title.set_text('velocity={:.2f} $\mu$m/sec'.format(slope))


#ax2.plot(time_ar, disp_microns_ar, 'go', ms=2)
#ax2.plot(time_ar, line, 'k--', label='velocity={:.2f} $\mu$m/sec'.format(slope))
#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., frameon=False)
#plt.pause(0.001)
	
#camera.snap()

animation = camera.animate()
animation.save('velocity.gif')

vidObj.release()
cv2.destroyAllWindows()

#plt.legend()
#plt.show()

