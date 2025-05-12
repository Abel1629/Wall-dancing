using UnityEngine;

// When the player makes collision with the walls
public class CollisionDetection : MonoBehaviour
{
    public float fallForce = 4000.0f; // this force is added to the player when it collides with a wall
    public new Camera camera; // camera object
    private PoseSwitcher poseSwitcher; // pose switcher component of the parent

    [Header ("----- Game managers -----")]
    [SerializeField] UIManager uiManager;
    [SerializeField] AudioManager audioManager;

    private void Awake()
    {
        audioManager = GameObject.FindGameObjectWithTag("AudioManager").GetComponent<AudioManager>();
    }

    private void Start()
    {
        poseSwitcher = GetComponentInParent<PoseSwitcher>();
    }

    // if it collides with the wall
    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Wall"))
        {
            Rigidbody rb = GetComponent<Rigidbody>();

            if (rb == null)
            {
                rb = gameObject.AddComponent<Rigidbody>();
            }

            rb.useGravity = true;

            rb.AddForce(0, 0, -fallForce * Time.deltaTime, ForceMode.Impulse);

            poseSwitcher.wallCollisionMade = true; // after changing it, there is no posibility to change the position

            // playing sound effects on collision
            audioManager.PlaySFX(audioManager.collision);
            audioManager.PlaySFX(audioManager.death);

            uiManager.GameOver();
        }
    }

    private void Update()
    {
        // when the player made collision, the camera will track it's falling
        if (poseSwitcher != null && poseSwitcher.wallCollisionMade)
        {
            camera.transform.LookAt(transform.position);
        }
    }

}
