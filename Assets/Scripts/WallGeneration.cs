using Unity.Mathematics;
using UnityEngine;

// It spawns new walls when the trigger interacts with the player
public class WallGeneration : MonoBehaviour
{
    public GameObject[] wallPrefabs;
    public float speedIncrement = 1f;  // Amount to increase speed each time
    public SetScore score; // to update the score
    private float currentSpeed; // Starting speed
    private System.Random random = new System.Random();

    private AudioManager audioManager;

    private void Awake()
    {
        audioManager = GameObject.FindGameObjectWithTag("AudioManager").GetComponent<AudioManager>();
    }

    public void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("WallTrigger"))
        {
            HasTriggered hasTriggered = other.GetComponent<HasTriggered>();

            if (hasTriggered != null && !hasTriggered.triggered)
            {
                currentSpeed = other.GetComponentInParent<Move>().speed; // speed of the current wall

                score.scoreValue++; // increasing the score

                // playing sound effect for scoring
                audioManager.PlaySFX(audioManager.score);

                // creating a random number, to generate the walls in random order
                int randomWallIndex = random.Next(0, wallPrefabs.Length);

                if (wallPrefabs[randomWallIndex] == null)
                {
                    Debug.LogWarning("Selected wallPrefab is null or has been destroyed!");
                    return;
                }

                // Instantiate the wall
                GameObject newWall;

                // these two walls need to start on a different Y level
                if (randomWallIndex == 2 || randomWallIndex == 12)
                {
                    Vector3 startingPosition = new Vector3(0f, 4f, 120f);
                    newWall = Instantiate(wallPrefabs[randomWallIndex], startingPosition, Quaternion.identity);
                }

                else
                {
                    Vector3 startingPosition = new Vector3(0f, 4.50885f, 120f);
                    newWall = Instantiate(wallPrefabs[randomWallIndex], startingPosition, Quaternion.identity);
                }

                newWall.gameObject.SetActive(true);

                Debug.Log(newWall.name + " created with random number " + randomWallIndex);

                //Set the new speed
                Move moveScript = newWall.GetComponent<Move>();
                if (moveScript != null)
                {
                    moveScript.speed = currentSpeed + 2 * speedIncrement; // increasing the speed of the new wall
                                                                          // *2 to make it work smoother
                }

                Debug.Log("Wall speed: " + moveScript.speed);

                hasTriggered.ChangeTriggerState();
            }
        }
    }
}
