
D=0.01 #aperture diameter 10mm =0.01
f=0.05 #focal length (brennvidde) 50mm = 0.05m
s=5    #distance from length 5m
lambd=500*1e-9 #wavelength 500 nanometres

#calculate minimum distance using reigligh criteria:
min_l= (1.22*s*lambd)/D
print('minimum distance using reigligh in mm:', min_l*1000)

#distance in picture plane (bildeplanet)
min_pic_l = min_l*f/(s-f)
print('minimum distance using reigligh in micro metre:', min_pic_l*1000*1000)

#limit for minimum period to maximum frequecy
T = min_pic_l #period
f = 1/T #frequency
print('period (T):',T, 'frequency:', f)

#samplings theory states tht we should have 2 samples per period, so: T/2 = 1.54 micrometre

#size of CDDis at 16mmx24mm how many pixel elements are needed?
"""
16mm --> 16/1.54
24mm --> 24/1.54
therefore: 10400 x 156000 elements
162 mega samples
"""

#Om aparturediameteren dobles, ville den romlige oppløsningen bli bedre eller dårligere?
#bedre fordi minste avstand for å skille to punkter vil bli mindre