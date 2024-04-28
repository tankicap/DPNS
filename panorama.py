import cv2
import argparse

p=argparse.ArgumentParser(description='Stitching images')
p.add_argument('image_paths', nargs='+', help='paths to images')
args=p.parse_args()

imgs=[]
for path in args.image_paths:
    image = cv2.imread(path)
    if image is None:
        print(f"Failed to load image: {path}")
        exit(1)
    imgs.append(image)

desired_width=400
resized_images=[]
for img in imgs:
    ratio=desired_width/img.shape[1]
    desiredHheight=int(img.shape[0] * ratio)
    resizedImg=cv2.resize(img, (desired_width, desiredHheight))
    resized_images.append(resizedImg)

grayscaleImgs=[cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in resized_images]

sift = cv2.SIFT_create()
keypoints=[]
descriptors=[]
for img in grayscaleImgs:
    kp,desc=sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(desc)

bf=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches=[]
for i in range(len(descriptors) - 1):
    matches.append(bf.match(descriptors[i], descriptors[i+1]))

sortedMatches=[sorted(match, key=lambda x: x.distance) for match in matches]

N=50
matchedImgs=[]
for i in range(len(sortedMatches)):
    matchedImg = cv2.drawMatches(resized_images[i], keypoints[i], resized_images[i+1], keypoints[i+1],
                                    sortedMatches[i][:N], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matchedImgs.append(matchedImg)

stitcher=cv2.Stitcher.create()
status,result=stitcher.stitch(resized_images)

cv2.imshow('Result', result)
for i, matchedImg in enumerate(matchedImgs):
    cv2.imshow(f'Keypoint Matches {i+1}-{i+2}', matchedImg)

cv2.waitKey(0)
cv2.destroyAllWindows()