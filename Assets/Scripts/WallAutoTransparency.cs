using UnityEngine;

// when the wall passes a point, it will transform into a transparent one
public class WallAutoTransparency : MonoBehaviour
{
    public float zThreshold = 0f; // This is the point where the wall will transform it's material to transparent
    public Material transparentMaterial; // transparent material
    public Material solidMaterial; // solid material (base)

    private bool isTransparent = false;
    private Renderer wallRenderer;

    private void Start()
    {
        wallRenderer = GetComponent<Renderer>();
        if (wallRenderer != null)
        {
            wallRenderer.material = solidMaterial; // at start the material is solid
        }
    }

    void Update()
    {
        if (!isTransparent && transform.position.z < zThreshold) // when it passes the treshold, it transformes
        {
            if (wallRenderer != null)
            {
                wallRenderer.material = transparentMaterial;
            }
            isTransparent = true;
        }
    }
}
