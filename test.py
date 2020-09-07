try:
    import medical_volume_slicer as mvs

    print("Medical Volume Slicer Successfully installed and imported.")

except ModuleNotFoundError:
    print("Medical Volume Silcer Not install correctly")
    print(
        "Consider using `python setup.py develop` to make this package install as a link"
    )
except:
    raise

from medical_volume_slicer.core import Volume

v = Volume(R"D:\CITI\kits19\data\case_00000\imaging.nii.gz")

print(v.get_yield_slice('sagittal')[25])