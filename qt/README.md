# CT Editor Graphical Interface
A GUI created to interact with the trained network post-training.

## Dependencies

-[PyQT5](https://www.riverbankcomputing.com/static/Docs/PyQt5/): See [here](https://www.metachris.com/2016/03/how-to-install-qt56-pyqt5-virtualenv-python3/#install-sip) for install instructions.

-pydicom: `pip install pydicom`

-SimpleITK: `pip install SimpleITK`

-scikit image: `pip install scikit-image`


## Editing pre-exisiting CT scans to have tumours inserted in hand-chosen locations.
                                   
## Before Starting

Move your CT scan (preferably in dicom format, although .mhd files work for part of the GUI) into the [ct_scans directory](../data/ct_scans).
    - If you are hoping to create a new dicom file with the structures, have your files formatted similar to the below path layout. Note that the RS.dcm file is the pre-existing RT structure file.
    
<p align="left">
  <img width="1000" height="350" src="../figures/file_format.png">
</p>

## Editing your CT

1. Open your Terminal and move to the [CT_Editor/qt directory](.).

2. Run the command `python main.py` and you will see a window open.

3. Locate your CT scan and select `Load Scan`.

4. Another window will open allowing you to select a location (by selecting a location within the scan) and size of the cube (by using the scroll box labelled "Cube Side Length") that you would like to edit. 

    - Note that the selection of your location will not work if you have the zoom button (the magnifying glass) already selected; to correct for this simply press the zoom button again to deselect it.

5. Once you have suitable cube location and size, the `Apply Edit` button will run the network on the selected cube and insert the edited version. If you are unhappy with the edit, simply press `Undo Edit` to remove this edit and try a different location and/or size.

6. You can add more than one edit if you'd like, or if you are satisfied you can (1) move onto the [segmentation](#segmenting-your-tumour) of your inserted tumours or (2) save the edited CT image as an `.npz` file using the `Save Scan` button.

## Segmenting your tumour

1. In order to continue on with the segmentation of your inserted tumours, press the `Segment Tumour` button, which will open a new window.

2. From this new window, you can view the pre-exisiting RT structures by selecting one of the options under the `Current Structure` choices.

3. To create a new contour structure, type the name of your new structure in the box next to `Create Structure` and then press the `Create Structure` button.

4. Now when you press on the image slice, points will appear that represent your new contour points. To make this easier, utilize the zoom button (the magnifying glass), although see the note under Step 4 to ensure you are able to continue select contour points.

5. Once satisfied with your contours on each slice (scroll through to make sure), you can save this new scan along with the updated contour file using the `Save dcm` button. Here you will be prompted to create a `New Folder` that will enclose your dicom slices and the RT Structure file.


