using UnityEngine;


// This class adds accurate colliders for the bones of the player
public class AddHitboxColliders : MonoBehaviour
{
    // The size of each collider
    public float colliderRadius = 0.05f;
    public float colliderLength = 0.08f;

    void Start()
    {
        AddColliderToBone("mixamorig5:Hips");
        AddColliderToBone("mixamorig5:Spine");
        AddColliderToBone("mixamorig5:Spine1");
        AddColliderToBone("mixamorig5:Spine2");

        AddColliderToBone("mixamorig5:LeftUpperLeg");
        AddColliderToBone("mixamorig5:LeftLeg");
        AddColliderToBone("mixamorig5:LeftFoot");

        AddColliderToBone("mixamorig5:RightUpperLeg");
        AddColliderToBone("mixamorig5:RightLeg");
        AddColliderToBone("mixamorig5:RightFoot");

        AddColliderToBone("mixamorig5:LeftShoulder");
        AddColliderToBone("mixamorig5:LeftArm");
        AddColliderToBone("mixamorig5:LeftForeArm");
        AddColliderToBone("mixamorig5:LeftHand");

        AddColliderToBone("mixamorig5:RightShoulder");
        AddColliderToBone("mixamorig5:RightArm");
        AddColliderToBone("mixamorig5:RightForeArm");
        AddColliderToBone("mixamorig5:RightHand");

        AddColliderToBone("mixamorig5:Neck");
        AddColliderToBone("mixamorig5:Head");
    }

    // It adds a collider to one bone
    void AddColliderToBone(string boneName)
    {
        Transform bone = FindDeepChild(transform, boneName);
        if (bone == null)
        {
            return;
        }

        CapsuleCollider collider = bone.gameObject.AddComponent<CapsuleCollider>();

        // I changed a bit the collider for the foot
        if (bone.name.CompareTo("mixamorig5:LeftFoot") == 0 || bone.name.CompareTo("mixamorig5:RightFoot") == 0) 
        {
            collider.radius = colliderRadius;
            collider.height = colliderLength;
            collider.direction = 1; // Y-axis

            Vector3 center = collider.center;
            center.y = 0.07f;
            collider.center = center;

        }
        // All the other colliders
        else
        {
            collider.radius = colliderRadius;
            collider.height = colliderLength;
            collider.direction = 1; // Y-axis
        }
    }

    // Regular recursive search method
    Transform FindDeepChild(Transform parent, string name)
    {
        foreach (Transform child in parent)
        {
            if (child.name == name)
                return child;
            Transform result = FindDeepChild(child, name);
            if (result != null)
                return result;
        }
        return null;
    }
}
