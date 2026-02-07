import bpy
import os
import sys


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def main():
    if "--" not in sys.argv:
        raise RuntimeError("Expected args after -- : <input_fbx> <output_obj>")
    argv = sys.argv[sys.argv.index("--") + 1 :]
    if len(argv) != 2:
        raise RuntimeError("Usage: blender -b --python convert_fbx_to_obj.py -- <input_fbx> <output_obj>")

    input_fbx = argv[0]
    output_obj = argv[1]

    if not os.path.exists(input_fbx):
        raise FileNotFoundError(input_fbx)

    os.makedirs(os.path.dirname(output_obj), exist_ok=True)

    reset_scene()
    bpy.ops.import_scene.fbx(filepath=input_fbx)

    # Ensure all imported meshes are selected for export
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)

    bpy.ops.export_scene.obj(
        filepath=output_obj,
        use_selection=True,
        use_materials=True,
        path_mode="COPY",
    )


if __name__ == "__main__":
    main()
