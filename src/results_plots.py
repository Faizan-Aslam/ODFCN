import pandas as pd
import matplotlib.pyplot as plt


combined = pd.read_csv('combined_uncertainity.csv')
middle =  pd.read_csv('middle_uncertainity.csv')
south =  pd.read_csv('south_uncertainity.csv')
east =  pd.read_csv('east_uncertainity.csv')
ground_truth = pd.read_csv('west_uncertainity.csv')
three = pd.read_csv('three_uncertainity.csv')
two = pd.read_csv('two_uncertainity.csv')
all_east = pd.read_csv('all_east_uncertainity.csv')
print(combined)
print(middle)
print(ground_truth)
print(south)
print(ground_truth)
print(all_east)


x = [x for x in range(1,39)]

print(x)
combined_unc = [2]+combined['col'].tolist()
middle_unc =[2]+ middle['col'].tolist()
south_unc = [2]+south['col'].tolist()
east_unc = [2]+east['col'].tolist()
ground_truth_unc = [2]+ground_truth['col'].tolist()
three_unc = [2]+three['col'].tolist()
two_unc = [2]+two['col'].tolist()
all_east_unc = [2]+all_east['col'].tolist()

legends = ['multiple_sensors','one_sensor','two_sensors','three_sensors']

plt.plot(x,combined_unc,color = 'blue')
plt.plot(x,middle_unc,color = 'red')
plt.plot(x,two_unc,color = 'orange')
plt.plot(x,three_unc,color = 'yellow')
#plt.plot(x,all_east_unc,color = 'green')

plt.legend(legends)
plt.xlabel('Timesteps')
plt.ylabel('Uncertainity')
#plt.title('Position Uncertainities for different combinations of sensor measurements')

plt.show()



combined_x = [2]+combined['x_estimate'].tolist()
middle_x =[2]+ middle['x_estimate'].tolist()
south_x = [2]+south['x_estimate'].tolist()
east_x = [2]+east['x_estimate'].tolist()
ground_truth_x = [2]+ground_truth['x_estimate'].tolist()
three_unc_x = [2]+three['x_estimate'].tolist()
two_unc_x = [2]+two['x_estimate'].tolist()
all_east_unc_x = [2]+all_east['x_estimate'].tolist()

combined_y = [2]+combined['y_estimate'].tolist()
middle_y =[2]+ middle['y_estimate'].tolist()
south_y = [2]+south['y_estimate'].tolist()
east_y = [2]+east['y_estimate'].tolist()
ground_truth_y = [2]+ground_truth['y_estimate'].tolist()
three_unc_y = [2]+three['y_estimate'].tolist()
two_unc_y = [2]+two['y_estimate'].tolist()
all_east_unc_y = [2]+all_east['y_estimate'].tolist()

def distances(ax,ay,bx,by):
    errors = []
    for i in range(0,37):
        print((ax[i],ay[i]),(bx[i],by[i]))
        distance = abs(ax[i]-bx[i])+abs(ay[i]-by[i])
        errors.append(distance)
    return errors

combined_error = distances(combined_x,combined_y,ground_truth_x,ground_truth_y)
middle_error = distances(middle_x,middle_y,ground_truth_x,ground_truth_y)
two_error = distances(two_unc_x,two_unc_y,ground_truth_x,ground_truth_y)
three_error = distances(three_unc_x,three_unc_y,ground_truth_x,ground_truth_y)
east_error = distances(east_x,east_y,ground_truth_x,ground_truth_y)
all_east_error = distances(all_east_unc_x,all_east_unc_y,ground_truth_x,ground_truth_y)

legends = ['multiple_sensors','one_sensor','two_sensors','three_sensors','all_east']

x = [x for x in range(1,38)]
plt.plot(x,combined_error,color = 'blue')
plt.plot(x,east_error,color = 'red')
plt.plot(x,two_error,color = 'orange')
plt.plot(x,three_error,color = 'yellow')
plt.plot(x,all_east_error,color = 'green')

plt.legend(legends)
plt.xlabel('Timesteps')
plt.ylabel('error')
plt.title('Position error for different combinations of sensor measurements')
plt.show()