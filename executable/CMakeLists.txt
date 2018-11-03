
add_executable(assignment_learning_ex assignment_learning_ex.cpp)
target_link_libraries(assignment_learning_ex dlib::dlib)

add_executable(face_detection_ex_onefolder face_detection_ex_onefolder.cpp)
target_link_libraries(face_detection_ex_onefolder dlib::dlib ${OpenCV_LIBS})

add_executable(face_detection_ex_oneimg face_detection_ex_oneimg.cpp)
target_link_libraries(face_detection_ex_oneimg dlib::dlib ${OpenCV_LIBS})

#landmark
add_executable(face_landmark_detection_ex face_landmark_detection_ex.cpp)
target_link_libraries(face_landmark_detection_ex dlib::dlib ${OpenCV_LIBS})

add_executable(dnn_face_recognition_ex dnn_face_recognition_ex.cpp)
target_link_libraries(dnn_face_recognition_ex dlib::dlib ${OpenCV_LIBS})

#---------------------------------------------------------------------
# wrapper dlib to a shared lib and use opencv to read image
add_executable(detectface detectface.cpp)
target_link_libraries(detectface readparameter dlibfacedetection dlib::dlib ${OpenCV_LIBS})

add_executable(deepfeatureextration deepfeatureextration.cpp)
target_link_libraries(deepfeatureextration readparameter dlibdeepfaceprocess dlibfacedetection dlib::dlib ${OpenCV_LIBS})
#---------------------------------------------------------------------
# call camera and make face pool
add_executable(pufa_CameraDetectandMakeFacePool pufa_CameraDetectandMakeFacePool.cpp)
target_link_libraries(pufa_CameraDetectandMakeFacePool  myfUtils readparameter dlibfacedetection dlibdeepfaceprocess dlibdeepfacedescriptormanager dlib::dlib ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})

add_executable(pufa_EncodeFacePool pufa_EncodeFacePool.cpp)
target_link_libraries(pufa_EncodeFacePool myfUtils readparameter dlibfacedetection dlibdeepfaceprocess dlibdeepfacedescriptormanager dlib::dlib ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})

add_executable(pufa_DetectFaceandVerify pufa_DetectFaceandVerify.cpp)
target_link_libraries(pufa_DetectFaceandVerify myfUtils readparameter dlibfacedetection dlibdeepfaceprocess dlibdeepfacedescriptormanager dlib::dlib ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})



#---------------------------------------------------------------------
# 在准备好的熟人，陌生人库上测试识别准确率
add_executable(test_recognition_accuracy_ex test_recognition_accuracy_ex.cpp)
target_link_libraries(test_recognition_accuracy_ex myfUtils readparameter dlibfacedetection dlibdeepfaceprocess dlibdeepfacedescriptormanager dlib::dlib ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})

