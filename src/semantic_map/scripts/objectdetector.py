#!/usr/bin/env python3
import rospy
from detection_msgs.msg import BoundingBoxes
from geometry_msgs.msg import PoseStamped,Point32
from sensor_msgs.msg import PointCloud,PointCloud2,ChannelFloat32,PointField
import sensor_msgs.point_cloud2 as pc2
import cv2
import numpy as np
import message_filters
from scipy import stats
import struct
import csv
import os
import math

def pointcloud2_to_points(msg):   #Converts PointCloud2 to PointCloud to simplify Distance calcultion
    cloud_points = []
    for point in pc2.read_points(msg, skip_nans=True):
        cloud_points.append(Point32(point[0], point[1], point[2]))
    return cloud_points

# def point_cloud2_to_points(point_cloud2):
#     points = []
#     point_fields = point_cloud2.fields

#     # get the offsets of the x, y, and z fields
#     x_offset = [field.offset for field in point_fields if field.name == 'x'][0]
#     y_offset = [field.offset for field in point_fields if field.name == 'y'][0]
#     z_offset = [field.offset for field in point_fields if field.name == 'z'][0]
#     # calculate the point step size in bytes
#     point_step = point_cloud2.point_step

#     # iterate over the byte array in the data field, extracting x, y, and z values
#     for i in range(0, len(point_cloud2.data), point_step):
#         x = get_field_value(point_cloud2.data, x_offset, point_fields[0].datatype)
#         y = get_field_value(point_cloud2.data, y_offset, point_fields[1].datatype)
#         z = get_field_value(point_cloud2.data, z_offset, point_fields[2].datatype)
#         # create a new Point message and append it to the list
#         point = Point32(x=x, y=y, z=z)
#         points.append(point)
#     return points

# def get_field_value(data, offset, datatype):
#     if datatype == PointField.FLOAT32:
#         gg=struct.unpack('f', data[offset:offset+4])[0]
#         if not math.isnan(gg):
#          print(gg)
#         return gg
#     elif datatype == PointField.FLOAT64:
#         return struct.unpack('d', data[offset:offset+8])[0]
#     elif datatype == PointField.INT8:
#         return struct.unpack('b', data[offset:offset+1])[0]
#     elif datatype == PointField.UINT8:
#         return struct.unpack('B', data[offset:offset+1])[0]
#     elif datatype == PointField.INT16:
#         return struct.unpack('h', data[offset:offset+2])[0]
#     elif datatype == PointField.UINT16:
#         return struct.unpack('H', data[offset:offset+2])[0]
#     elif datatype == PointField.INT32:
#         return struct.unpack('i', data[offset:offset+4])[0]
#     elif datatype == PointField.UINT32:
#         return struct.unpack('I', data[offset:offset+4])[0]
#     elif datatype == PointField.INT64:
#         return struct.unpack('q', data[offset:offset+8])[0]
#     elif datatype == PointField.UINT64:
#         return struct.unpack('Q', data[offset:offset+8])[0]
#     else:
#         raise ValueError('Unknown PointField datatype: {}'.format(datatype))

class obj_position():
     def __init__(self):
         self.prev_cam=np.array([0,0,0])    
         # camera information
         self.im_size_x = 768  #720 for habitat
         self.im_size_y = 492
         self.K = np.array([[407.0646129842357, 0, 384.5],
                            [0, 407.0646129842357, 246.5],
                            [0,      0   ,     1    ]])  # camera K matrix

         self.founded_objects = PointCloud()
         self.object_class = ChannelFloat32()

         rospy.init_node('listener', anonymous=True)
         self.bb_sub = message_filters.Subscriber('/yolov5/detections', BoundingBoxes)
         self.cam_sub = message_filters.Subscriber('/rexrov/ground_truth_to_tf_rexrov/pose', PoseStamped) #the pose is the realtime localised position of the rov
         self.pc_sub = message_filters.Subscriber('/rexrov/rexrov/camera/camera_cloud', PointCloud2)# needs to be replaced with ORB-SLAM point cloud 
         self.obj_pub = rospy.Publisher('/obj_position', PointCloud , queue_size=1)         
         self.ts = message_filters.ApproximateTimeSynchronizer([self.bb_sub,self.cam_sub,self.pc_sub],1,1) #boundingbox is calculated when pointcloud matching with yolo bounding box
         self.ts.registerCallback(self.gotdata)         



     def gotdata(self,bound,cam_pose,pc):
         points_cat = []
         points_cat_mid = []
         objects = []
         objects_info = []

         c=np.zeros(3)
         R=np.zeros(4)
         
         c[0] = cam_pose.pose.position.x
         c[1] = cam_pose.pose.position.y
         c[2] = cam_pose.pose.position.z

         R[0] = cam_pose.pose.orientation.x
         R[1] = cam_pose.pose.orientation.y
         R[2] = cam_pose.pose.orientation.z
         R[3] = cam_pose.pose.orientation.w
         
         with open('/home/nivnoetic/uuv_ws/src/projection/projection.csv', mode='a') as proj_file:
             proj_writer = csv.writer(proj_file, delimiter=',')
             proj_writer.writerow([c[0],c[1],c[2],R[0],R[1],R[2],R[3]])        #store values in a csv file

         K = self.K 
         Rot = np.array([[1-2*(R[1]**2)-2*(R[2]**2),2*R[0]*R[1]+2*R[2]*R[3],2*R[0]*R[2]-2*R[1]*R[3],0],
                         [2*R[0]*R[1]-2*R[2]*R[3],1-2*(R[0]**2)-2*(R[2]**2),2*R[1]*R[2]+2*R[0]*R[3],0],
                         [2*R[0]*R[2]+2*R[1]*R[3],2*R[1]*R[2]-2*R[0]*R[3],1-2*(R[0]**2)-2*(R[1]**2),0],
                         [0,0,0,1]])         

         Proj = np.zeros((3,4))       #Projecting camera image to match with object sizes
         Proj[0:3,0:3] = Rot[0:3,0:3]
         tcw_ros = np.array([[-c[0]],[-c[1]],[-c[2]]]) # change axis of view
         tcw_orb = np.dot(Rot[0:3,0:3],tcw_ros)

         Proj[0,3]=tcw_orb[0]
         Proj[1,3]=tcw_orb[1]
         Proj[2,3]=tcw_orb[2]          
         cur_cam = cam_pose.pose.position
         
         self.Projection = np.dot(K,Proj)
         d=np.array([self.prev_cam[0]-cur_cam.x,self.prev_cam[1]-cur_cam.y,self.prev_cam[2]-cur_cam.z])
         self.prev_cam = np.array([cur_cam.x,cur_cam.y,cur_cam.z])
         geo_points = pointcloud2_to_points(pc)
         #channels = pc.channels
         obj = 0

         for bb in bound.bounding_boxes:
              current_time = rospy.Time.now()
              ins = []
              xmin = bb.xmin
              xmax = bb.xmax
              ymin = bb.ymin
              ymax = bb.ymax
              Class = bb.Class
              id_obj = bb.id # object id given by classifier
              #id_obj= 22
              prob = bb.probability
              if Class == "None":
                   obj = obj - 1
                   continue
              obj = obj + 1
              xmid = (xmin+xmax)/2
              ymid = (ymin+ymax)/2
              print("I saw a "+Class)
              if xmin < self.im_size_x/20 or xmax > 24*self.im_size_x/25   or ymin < self.im_size_y/20:  #to exclude objects that are too close to the camera
                   continue          
              print("considered "+Class)
              for i in range(len(geo_points)):
                   fx = geo_points[i].x
                   fy = geo_points[i].y
                   fz = geo_points[i].z
                   pt3d = np.array([[fx],[fy],[fz],[1]])
                #    if not math.isnan(geo_points[i].y):
                #     print("pt3d=",fx,fy,fz,"xmax",xmax,"xmin",xmin,"ymax",ymax)
                   pt2d = np.dot(self.Projection,pt3d)
                   pt2d[0] = pt2d[0] / pt2d[2]
                   pt2d[1] = pt2d[1] / pt2d[2]
                   pt2d[2] = pt2d[2] / pt2d[2]
                   if pt2d[0]<xmax and xmin<pt2d[0] and pt2d[1]<ymax and ymin<pt2d[1]:
                        dis_p_v = np.array([xmid - pt2d[0],ymin - pt2d[1]])
                        dis_p = np.linalg.norm(dis_p_v)
                        in_bb = [dis_p,fx,fy,fz]
                        ins.append(in_bb)

              ins = np.array(ins)
              if len(ins)>0:
                  median_z = np.median(ins[:,3])  # z is forward
                  for i in range(len(ins)-1,-1,-1):
                      if (ins[i,3]-median_z) > 0.5:
                          ins = np.delete(ins,i,axis=0)                  
              if len(ins) == 0:
                   continue
              ins = ins[ins[:,0].argsort()]
              if len(ins)>3:
                   closests = ins[0:3]
              else:
                   closests = ins
              closests = np.array(closests)

              print("closests are\n"+str(closests))

              for j in range(len(closests)-1,0,-1):  #consider close points to the object the same
                  dc_v = [closests[j,1]-closests[0,1] ,closests[j,2]-closests[0,2], closests[j,3]-closests[0,3]]
                  dc = np.linalg.norm(dc_v)
                  if dc>1:
                      closests = np.delete(closests,j,axis=0)

              for j in range(len(closests)-1,-1,-1):
                  if closests[j,0] > 30: 
                      closests = np.delete(closests,j,axis=0) 
              if len(closests) == 0:
                  continue

              Xs = closests[:,1]
              Ys = closests[:,2]
              Zs = closests[:,3]
              X = np.mean(Xs)
              Y = np.mean(Ys)
              Z = np.mean(Zs)
              points = [X,Y,Z,Class,obj,id_obj,xmid,ymid,xmin,xmax,ymin,ymax]
              points_cat.append(points)

         points_cl = np.array(points_cat)
         for i in range(len(points_cl)):
              if(len(points_cl)==0):
                   break                #creating object point cloud

              print("saw a "+points_cl[i,3]+ " at: "+str([float(points_cl[i,0]),float(points_cl[i,1]),float(points_cl[i,2])]))
              object_position = Point32()
              object_position.x = float(points_cl[i,0])
              object_position.y = float(points_cl[i,1])
              object_position.z = float(points_cl[i,2])
              objects.append(object_position)
              objects_info.append(float(points_cl[i,5]))
              objects_info.append(float(points_cl[i,8]))
              objects_info.append(float(points_cl[i,9]))
              objects_info.append(float(points_cl[i,10]))
              objects_info.append(float(points_cl[i,11]))
              self.object_class.values = np.array(objects_info)
              self.object_class.name = "objects"

         self.founded_objects.points = objects
         self.founded_objects.channels = [self.object_class]
         self.founded_objects.header.stamp = rospy.Time.now()
         self.founded_objects.header.frame_id = "rexrov/camera_link_optical"
         self.obj_pub.publish(self.founded_objects)
         
if __name__ == '__main__':
    path_file = '/home/nivnoetic/uuv_ws/src/projection/projection.csv'
    if os.path.exists(path_file):
        os.remove(path_file)
    print("init")
    obj_position()
    rospy.spin()