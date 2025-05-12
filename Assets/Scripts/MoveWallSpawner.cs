using UnityEngine;

// it creates a trigger which is moving with the wall
// when this trigger passes the player it will create new walls
public class MoveWallSpawner : MonoBehaviour
{
    public float zOffset = 0f;

    private GameObject wallSpawner;

    void Start()
    {
        // Create a new GameObject named "WallSpawner"
        wallSpawner = new GameObject("WallSpawner");

        // Set its initial position relative to the Wall (this GameObject)
        wallSpawner.transform.position = new Vector3(0f, 1f, transform.position.z + zOffset);

        // Set the parent (optional for hierarchy organization)
        wallSpawner.transform.parent = this.transform;

        // Set its scale
        wallSpawner.transform.localScale = new Vector3(15f, 15f, 1f);

        // Add a BoxCollider and set it to be a trigger
        BoxCollider collider = wallSpawner.AddComponent<BoxCollider>();
        collider.isTrigger = true;

        // Add the HasTriggered script
        wallSpawner.AddComponent<HasTriggered>();

        // Set tag to "WallTrigger"
        wallSpawner.tag = "WallTrigger";
    }

    void Update()
    {
        if (wallSpawner != null)
        {
            // Keep updating the WallSpawner's position based on the wall's current Z
            wallSpawner.transform.position = new Vector3(0f, 1f, transform.position.z + zOffset);
        }
    }
}
