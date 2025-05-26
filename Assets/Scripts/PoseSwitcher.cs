using UnityEngine;

public class PoseSwitcher : MonoBehaviour
{
    public int currentPoseIndex = 0;

    public bool wallCollisionMade = false;

    private Transform[] poses;

    [SerializeField] public string pose = "0";

    void Start()
    {
        // the total number of poses
        int childCount = transform.childCount;

        // creating an array which will contain every pose
        poses = new Transform[childCount];

        // Get all poses from the children
        for (int i = 0; i < childCount; i++)
        {
            poses[i] = transform.GetChild(i);
            poses[i].gameObject.SetActive(false);
        }

        // Activate the base pose
        if (poses.Length > 0)
        {
            poses[0].gameObject.SetActive(true);
        }
    }

    void Update()
    {
        SwitchToPose(pose);
    }

    // makes the pose on the given index active, also deactivates the previous pose
    public void Switching(int index)
    {
        poses[currentPoseIndex].gameObject.SetActive(false);
        poses[index].gameObject.SetActive(true);
        currentPoseIndex = index;
    }

    // changes to a pose based on the input
    public void SwitchToPose(string pose)
    {
        if (!wallCollisionMade) {
            switch (pose)
            {
                case "0":
                    Switching(0);
                    break;

                case "1":
                    Switching(1);
                    break;

                case "2":
                    Switching(2);
                    break;

                case "3":
                    Switching(3);
                    break;

                case "4":
                    Switching(4);
                    break;

                case "5":
                    Switching(5);
                    break;

                case "6":
                    Switching(6);
                    break;

                case "7":
                    Switching(7);
                    break;

                case "8":
                    Switching(8);
                    break;

                case "9":
                    Switching(9);
                    break;

                case "10":
                    Switching(10);
                    break;

                case "11":
                    Switching(11);
                    break;

                case "12":
                    Switching(12);
                    break;
            }
        }
        
    }
}

