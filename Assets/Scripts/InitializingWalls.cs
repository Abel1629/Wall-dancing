using UnityEngine;

// it initializes two walls at the start
public class InitializingWalls : MonoBehaviour
{
    public GameObject[] wallPrefabs;
    private System.Random random = new System.Random();

    void Start()
    {
        for (int i = 1; i <= 2; i++) 
        {
            // creating a random number, to generate the walls in random order
            int randomWallIndex = random.Next(0, wallPrefabs.Length);

            // Instantiate the wall
            GameObject newWall;

            if (randomWallIndex == 2 || randomWallIndex == 12)
            {
                Vector3 startingPosition = new Vector3(0f, 4f, 60f * i);
                newWall = Instantiate(wallPrefabs[randomWallIndex], startingPosition, Quaternion.identity);
            }

            else
            {
                Vector3 startingPosition = new Vector3(0f, 4.50885f, 60f * i);
                newWall = Instantiate(wallPrefabs[randomWallIndex], startingPosition, Quaternion.identity);
            }

            newWall.gameObject.SetActive(true);

            if (i == 2)
            {
                //Set a higher speed for the second wall
                Move moveScript = newWall.GetComponent<Move>();

                if (moveScript != null)
                {
                    moveScript.speed += 1; // increasing the speed of the new wall
                }
            }
        }
    }

}
